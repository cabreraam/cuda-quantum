/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/UnitaryMeasurementAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <iterator>

namespace cudaq::opt {
#define GEN_PASS_DEF_GROUPUNITARIESANDMSMTSSLICES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "group-unitaries-and-msmts-slices"

using namespace mlir;
using cudaq::quake::detail::UnitaryMeasurementInfo;

namespace {
struct ProgramSliceOp {
  ProgramSliceOp() = default;
  ProgramSliceOp(Operation *op,
                 const llvm::SetVector<Value> &controlDependencies)
      : op(op), controlDependencies(controlDependencies) {}

  Operation *op = nullptr;
  llvm::SetVector<Value> controlDependencies;
};

struct ProgramSlice {
  explicit ProgramSlice(Operation *measurement) : measurement(measurement) {}

  Operation *measurement = nullptr;
  SmallVector<ProgramSliceOp, 16> ops;
  SmallVector<Operation *> unitaryOps;
  SmallVector<Operation *> measurementOps;
  llvm::SetVector<Value> dependencyValues;
  llvm::SetVector<Value> controlDependencyValues;
};

static bool isUnitary(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() && !isa<cudaq::quake::ResetOp>(op);
}

static bool isMeasurement(Operation *op) {
  return op->hasTrait<cudaq::QuantumMeasure>();
}

static SmallVector<Value> getOperands(Operation *op) {
  return SmallVector<Value>(op->operand_begin(), op->operand_end());
}

static SmallVector<Value> getResults(Operation *op) {
  return SmallVector<Value>(op->result_begin(), op->result_end());
}

/// Quantum operands/results are special because gates mutate quantum state
/// through operands rather than new SSA results.
static SmallVector<Value> getQuantumOperands(Operation *op) {
  SmallVector<Value> values;
  for (auto operand : op->getOperands())
    if (cudaq::quake::isQuantumType(operand.getType()))
      values.push_back(operand);
  return values;
}

static SmallVector<Value> getQuantumResults(Operation *op) {
  SmallVector<Value> values;
  for (auto result : op->getResults())
    if (cudaq::quake::isQuantumType(result.getType()))
      values.push_back(result);
  return values;
}

static bool containsAny(const llvm::SetVector<Value> &set,
                        ArrayRef<Value> values) {
  for (auto value : values)
    if (set.contains(value))
      return true;
  return false;
}

static void insertAll(llvm::SetVector<Value> &set, ArrayRef<Value> values) {
  for (auto value : values)
    set.insert(value);
}

static void insertAll(llvm::SetVector<Value> &set,
                      const llvm::SetVector<Value> &values) {
  for (auto value : values)
    set.insert(value);
}

static void removeAll(llvm::SetVector<Value> &set, ArrayRef<Value> values) {
  for (auto value : values)
    set.remove(value);
}

/// If `op` was inside a `cc.if`, add its condition values to the slice and
/// frontier so their producers can be found.
static void addControlDependencies(ProgramSlice &slice,
                                   llvm::SetVector<Value> &frontier,
                                   Operation *op,
                                   const UnitaryMeasurementInfo &info) {
  auto iter = info.opControlDependencies.find(op);
  if (iter == info.opControlDependencies.end())
    return;
  insertAll(frontier, iter->second);
  insertAll(slice.dependencyValues, iter->second);
  insertAll(slice.controlDependencyValues, iter->second);
}

static void addSliceOp(ProgramSlice &slice, Operation *op,
                       const UnitaryMeasurementInfo &info) {
  auto iter = info.opControlDependencies.find(op);
  if (iter == info.opControlDependencies.end())
    slice.ops.emplace_back(op, llvm::SetVector<Value>{});
  else
    slice.ops.emplace_back(op, iter->second);

  if (isUnitary(op))
    slice.unitaryOps.push_back(op);
  else if (isMeasurement(op))
    slice.measurementOps.push_back(op);
}

static ProgramSlice computeSlice(Operation *measurement,
                                 const UnitaryMeasurementInfo &info) {
  ProgramSlice slice(measurement);
  llvm::SetVector<Operation *> sliceOps;
  // Values whose defining/touching operations still need to be found.
  llvm::SetVector<Value> frontier;

  // Seed the slice with the anchor measurement and the values it immediately
  // reads/defines.
  sliceOps.insert(measurement);
  insertAll(frontier, getOperands(measurement));
  insertAll(slice.dependencyValues, getOperands(measurement));
  insertAll(slice.dependencyValues, getResults(measurement));
  addControlDependencies(slice, frontier, measurement, info);

  auto measurementIt = llvm::find_if(
      info.opOrder, [&](Operation *op) { return op == measurement; });
  assert(measurementIt != info.opOrder.end() &&
         "measurement must be in operation order");

  // Walk only the operations that occur before the measurement. An operation is
  // included if it defines an active SSA value or touches an active quantum
  // value. The quantum check is what pulls in earlier gates on the same qubit.
  for (auto it = std::make_reverse_iterator(measurementIt);
       it != info.opOrder.rend(); ++it) {
    Operation *op = *it;
    auto operands = getOperands(op);
    auto results = getResults(op);
    auto quantumOperands = getQuantumOperands(op);
    auto quantumResults = getQuantumResults(op);
    bool definesActiveValue = containsAny(frontier, results);
    bool touchesActiveQuantum = containsAny(frontier, quantumOperands) ||
                                containsAny(frontier, quantumResults);
    if (!definesActiveValue && !touchesActiveQuantum)
      continue;

    sliceOps.insert(op);
    removeAll(frontier, results);
    insertAll(frontier, operands);
    insertAll(slice.dependencyValues, operands);
    insertAll(slice.dependencyValues, results);
    addControlDependencies(slice, frontier, op, info);

    if (isa<cudaq::quake::ResetOp>(op)) {
      // Reset cuts off prior quantum history for that value. Keep the reset in
      // the slice, but stop chasing earlier gates on the same quantum state.
      removeAll(frontier, quantumOperands);
      removeAll(frontier, quantumResults);
    }
  }

  // Re-emit the collected operations in source order so debug output reads like
  // the original function, even though discovery happened backward.
  for (auto op : info.opOrder)
    if (sliceOps.contains(op))
      addSliceOp(slice, op, info);
  return slice;
}

static SmallVector<ProgramSlice, 4>
computeSlices(const UnitaryMeasurementInfo &info) {
  SmallVector<ProgramSlice, 4> slices;
  for (auto measurement : info.measurementOps)
    slices.push_back(computeSlice(measurement, info));
  return slices;
}

static void dumpSlice(unsigned index, const ProgramSlice &slice) {
  LLVM_DEBUG(llvm::dbgs() << "Measurement-centered slice " << index << ":\n");
  LLVM_DEBUG(llvm::dbgs() << "  Anchor: " << *slice.measurement << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Dependency values:");
  for (auto value : slice.dependencyValues)
    LLVM_DEBUG(llvm::dbgs() << "\n    " << value);
  LLVM_DEBUG(llvm::dbgs() << "\n  Control dependency values:");
  for (auto value : slice.controlDependencyValues)
    LLVM_DEBUG(llvm::dbgs() << "\n    " << value);
  LLVM_DEBUG(llvm::dbgs() << "\n");

  for (const ProgramSliceOp &sliceOp : slice.ops) {
    LLVM_DEBUG(llvm::dbgs() << "  Slice op: " << *sliceOp.op << "\n");
    if (!sliceOp.controlDependencies.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "    Control dependencies:");
      for (auto value : sliceOp.controlDependencies)
        LLVM_DEBUG(llvm::dbgs() << "\n     " << value);
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }
  }
}

struct GroupUnitariesAndMsmtsSlicesPass
    : public cudaq::opt::impl::GroupUnitariesAndMsmtsSlicesBase<
          GroupUnitariesAndMsmtsSlicesPass> {
  using GroupUnitariesAndMsmtsSlicesBase::GroupUnitariesAndMsmtsSlicesBase;

  void runOnOperation() override {
    auto func = getOperation();
    // Consume the classification/control-dependency analysis. This pass does
    // not require the analysis pass to run earlier in the pipeline; requesting
    // the analysis here is enough for MLIR to construct/cache it for this func.
    const auto &analysis =
        getAnalysis<cudaq::quake::detail::UnitaryMeasurementAnalysis>();
    const auto &analysisInfo = analysis.getAnalysisInfo();
    auto iter = analysisInfo.find(func);
    assert(iter != analysisInfo.end());

    LLVM_DEBUG(llvm::dbgs()
               << "\n\nProgram slices for: " << func.getName() << "\n");
    auto slices = computeSlices(iter->second);
    for (auto [index, slice] : llvm::enumerate(slices))
      dumpSlice(index, slice);
    LLVM_DEBUG(llvm::dbgs() << "\n");

    markAllAnalysesPreserved();
  }
};
} // namespace
