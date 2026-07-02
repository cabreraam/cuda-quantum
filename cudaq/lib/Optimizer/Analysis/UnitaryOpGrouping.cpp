/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <iterator>

#define DEBUG_TYPE "unitary-op-grouping-analysis"

using namespace mlir;

namespace {
/// Per-block state used while scanning clean value-semantics wire operations.
///
/// A phase identifies the current uninterrupted unitary segment for a wire.
/// Measurements advance only the measured wire or wires to a fresh phase.
struct BlockScanState {
  llvm::DenseMap<Value, unsigned> wireToPhase;
  llvm::DenseMap<unsigned, unsigned> phaseToGroupIndex;
  unsigned nextPhase = 1;

  void reset() {
    wireToPhase.clear();
    phaseToGroupIndex.clear();
    nextPhase = 1;
  }
};
} // namespace

/// Return true when \p op is a unitary quantum gate for grouping purposes.
static bool isUnitaryOp(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() && !isa<cudaq::quake::ResetOp>(op);
}

/// Return true when \p value is a scalar value-semantics wire.
static bool isWire(Value value) {
  return isa<cudaq::quake::WireType>(value.getType());
}

/// Return true when \p value has any Quake quantum type.
static bool hasQuantumValueType(Value value) {
  return cudaq::quake::isQuantumValueType(value.getType());
}

/// Return true when \p op consumes or produces any quantum value.
static bool hasAnyQuantumValue(Operation *op) {
  return llvm::any_of(op->getOperands(), hasQuantumValueType) ||
         llvm::any_of(op->getResults(), hasQuantumValueType);
}

/// Append \p value to \p values if it is not already present.
static void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

/// Append \p op to \p ops if it is not already present.
static void appendUnique(SmallVectorImpl<Operation *> &ops, Operation *op) {
  if (!llvm::is_contained(ops, op))
    ops.push_back(op);
}

/// Append scalar wire controls and targets from \p op to \p operands.
static void appendOperatorWireOperands(cudaq::quake::OperatorInterface op,
                                       SmallVectorImpl<Value> &operands) {
  for (Value control : op.getControls())
    if (isWire(control))
      operands.push_back(control);
  for (Value target : op.getTargets())
    if (isWire(target))
      operands.push_back(target);
}

/// Return the scalar wire controls and targets consumed by \p op.
static SmallVector<Value>
getOperatorWireOperands(cudaq::quake::OperatorInterface op) {
  SmallVector<Value> operands;
  appendOperatorWireOperands(op, operands);
  return operands;
}

/// Return true when \p op is a wire-only value-semantics unitary gate.
///
/// This deliberately excludes reference-semantics operations and aggregate
/// quantum values, which are handled by the conservative textual fallback.
static bool isCleanWireUnitaryOp(Operation *op) {
  if (!isUnitaryOp(op))
    return false;

  auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
  if (!gate)
    return false;

  if (!llvm::all_of(gate.getControls(), isWire) ||
      !llvm::all_of(gate.getTargets(), isWire))
    return false;

  SmallVector<Value> operands = getOperatorWireOperands(gate);
  ValueRange wires = gate.getWires();
  return !operands.empty() && operands.size() == wires.size() &&
         llvm::all_of(wires, isWire);
}

/// Return true when \p meas reads scalar wire targets and returns scalar
/// post-measurement wires.
static bool isCleanWireMeasurement(cudaq::quake::MeasurementInterface meas) {
  return !meas.getTargets().empty() &&
         llvm::all_of(meas.getTargets(), isWire) &&
         meas.getTargets().size() == meas.getWires().size() &&
         llvm::all_of(meas.getWires(), isWire);
}

/// Return the current value-semantics phase for \p value.
///
/// Unknown wires are treated as belonging to phase zero, allowing the first
/// clean wire unitary in a block to start the initial group.
static unsigned getPhaseFor(Value value, BlockScanState &state) {
  auto iter = state.wireToPhase.find(value);
  if (iter == state.wireToPhase.end())
    return 0;
  return iter->second;
}

/// Return the phase that should own \p gate.
///
/// Multi-wire gates join the latest phase among their wire operands, which
/// prevents a gate from being grouped before any measured predecessor wire.
static unsigned getPhaseForUnitaryOp(cudaq::quake::OperatorInterface gate,
                                     BlockScanState &state) {
  unsigned phase = 0;
  for (Value operand : getOperatorWireOperands(gate))
    phase = std::max(phase, getPhaseFor(operand, state));
  return phase;
}

unsigned
cudaq::quake::detail::UnitaryOpGroupingAnalysis::createGroup(Block &block) {
  unsigned groupIndex = static_cast<unsigned>(groups.size());

  UnitaryOpGroup group;
  group.block = &block;
  groups.push_back(std::move(group));
  blockToGroupIndices[&block].push_back(groupIndex);

  LLVM_DEBUG(llvm::dbgs() << "Created unitary group " << groupIndex << '\n');
  return groupIndex;
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::addOpToGroup(
    unsigned groupIndex, Operation *op) {
  UnitaryOpGroup &group = groups[groupIndex];
  appendUnique(group.ops, op);
  opToGroupIndex.try_emplace(op, groupIndex);

  if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op))
    for (Value wire : gate.getWires())
      if (isWire(wire))
        valueToGroupIndex.try_emplace(wire, groupIndex);
}

/// If we've hit this function, that means we've reached an op that cannot be
/// added to the current textual group of reference-semantics unitary ops. So,
/// we do one of two things.
/// 1) If the current group is empty, we don't do anything and just return.
/// 2) If the current group is non-empty, we create and populate a group, then
///    update the lookup tables for the new group.
void cudaq::quake::detail::UnitaryOpGroupingAnalysis::flushGroupIfNonEmpty(
    Block &block, mlir::SmallVectorImpl<Operation *> &currUnitaryOps) {
  if (currUnitaryOps.empty())
    return;

  unsigned groupIndex = createGroup(block);
  for (Operation *op : currUnitaryOps)
    addOpToGroup(groupIndex, op);

  LLVM_DEBUG(llvm::dbgs() << "Found textual unitary group with "
                          << currUnitaryOps.size() << " op(s)\n");

  currUnitaryOps.clear();
}

const mlir::Block *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getBlockForGroup(
    const UnitaryOpGroup &group) const {
  return group.block;
}

const cudaq::quake::detail::UnitaryOpGroup *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupContainingOp(
    mlir::Operation *op) const {
  if (!op)
    return nullptr;

  auto iter = opToGroupIndex.find(op);
  if (iter == opToGroupIndex.end())
    return nullptr;

  return &groups[iter->second];
}

const cudaq::quake::detail::UnitaryOpGroup *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupProducingValue(
    mlir::Value value) const {
  auto iter = valueToGroupIndex.find(value);
  if (iter == valueToGroupIndex.end())
    return nullptr;

  return &groups[iter->second];
}

mlir::SmallVector<const cudaq::quake::detail::UnitaryOpGroup *>
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupsIn(
    const mlir::Block *block) const {
  mlir::SmallVector<const UnitaryOpGroup *> groupsInBlock;
  if (!block)
    return groupsInBlock;

  auto iter = blockToGroupIndices.find(block);
  if (iter == blockToGroupIndices.end())
    return groupsInBlock;

  for (unsigned groupIndex : iter->second)
    groupsInBlock.push_back(&groups[groupIndex]);

  return groupsInBlock;
}

bool cudaq::quake::detail::UnitaryOpGroupingAnalysis::inSameGroup(
    mlir::Operation *op1, mlir::Operation *op2) const {
  if (!op1 || !op2)
    return false;

  auto group1 = opToGroupIndex.find(op1);
  if (group1 == opToGroupIndex.end())
    return false;

  auto group2 = opToGroupIndex.find(op2);
  if (group2 == opToGroupIndex.end())
    return false;

  return group1->second == group2->second;
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::
    populateGroupBoundaries() {
  for (UnitaryOpGroup &group : groups) {
    for (Operation *op : group.ops) {
      auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
      if (!gate)
        continue;

      for (Value operand : op->getOperands()) {
        if (isWire(operand)) {
          Operation *defOp = operand.getDefiningOp();
          if (!defOp || getGroupContainingOp(defOp) != &group)
            appendUnique(group.quantumInputs, operand);
          continue;
        }

        if (!isa<cudaq::quake::RefType, cudaq::quake::VeqType,
                 cudaq::quake::CableType, cudaq::quake::ControlType>(
                operand.getType())) {
          Operation *defOp = operand.getDefiningOp();
          if (!defOp || getGroupContainingOp(defOp) != &group)
            appendUnique(group.classicalInputs, operand);
        }
      }

      for (Value wire : gate.getWires()) {
        bool leavesGroup = wire.use_empty();
        for (Operation *user : wire.getUsers())
          if (getGroupContainingOp(user) != &group) {
            leavesGroup = true;
            break;
          }
        if (leavesGroup)
          appendUnique(group.quantumOutputs, wire);
      }
    }
  }
}

/// Scan a single block and form unitary groups. Reference-semantics gates keep
/// the legacy contiguous-run behavior. Clean wire-only value-semantics gates
/// are grouped by per-wire phases: a measurement advances only the measured
/// wire(s) to a new phase and is recorded as a readout boundary for the group
/// that produced the measured wire(s).
void cudaq::quake::detail::UnitaryOpGroupingAnalysis::scanBlock(Block &block) {
  mlir::SmallVector<Operation *> currTextualUnitaryOps;
  BlockScanState state;

  auto resetValueGrouping = [&]() { state.reset(); };

  for (Operation &op : block) {
    if (isCleanWireUnitaryOp(&op)) {
      flushGroupIfNonEmpty(block, currTextualUnitaryOps);

      auto gate = cast<cudaq::quake::OperatorInterface>(op);
      unsigned phase = getPhaseForUnitaryOp(gate, state);
      auto iter = state.phaseToGroupIndex.find(phase);
      unsigned groupIndex = iter == state.phaseToGroupIndex.end()
                                ? createGroup(block)
                                : iter->second;
      state.phaseToGroupIndex.try_emplace(phase, groupIndex);
      addOpToGroup(groupIndex, &op);

      for (Value wire : gate.getWires())
        state.wireToPhase[wire] = phase;
      continue;
    }

    if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
      flushGroupIfNonEmpty(block, currTextualUnitaryOps);

      if (isCleanWireMeasurement(meas)) {
        llvm::DenseMap<unsigned, unsigned> phaseToPostMeasurementPhase;
        SmallVector<Value> targets(meas.getTargets().begin(),
                                   meas.getTargets().end());
        SmallVector<Value> wires(meas.getWires().begin(),
                                 meas.getWires().end());

        for (auto [target, wire] : llvm::zip_equal(targets, wires)) {
          unsigned phase = getPhaseFor(target, state);
          auto groupIter = state.phaseToGroupIndex.find(phase);
          if (groupIter != state.phaseToGroupIndex.end()) {
            UnitaryOpGroup &group = groups[groupIter->second];
            auto readoutIter =
                llvm::find_if(group.readouts, [&](ReadoutBoundary &boundary) {
                  return boundary.measurement.getOperation() == &op;
                });
            if (readoutIter == group.readouts.end()) {
              ReadoutBoundary boundary;
              boundary.measurement = meas;
              group.readouts.push_back(std::move(boundary));
              readoutIter = std::prev(group.readouts.end());
            }
            appendUnique(readoutIter->measuredWires, target);
            appendUnique(readoutIter->postMeasurementWires, wire);
          }

          auto postPhaseIter = phaseToPostMeasurementPhase.find(phase);
          unsigned postMeasurementPhase;
          if (postPhaseIter == phaseToPostMeasurementPhase.end()) {
            postMeasurementPhase = state.nextPhase++;
            phaseToPostMeasurementPhase.try_emplace(phase,
                                                    postMeasurementPhase);
          } else {
            postMeasurementPhase = postPhaseIter->second;
          }
          state.wireToPhase[wire] = postMeasurementPhase;
        }
        continue;
      }

      resetValueGrouping();
      continue;
    }

    if (isUnitaryOp(&op)) {
      resetValueGrouping();
      currTextualUnitaryOps.push_back(&op);
      continue;
    }

    flushGroupIfNonEmpty(block, currTextualUnitaryOps);

    if (op.hasTrait<OpTrait::IsTerminator>() || op.getNumRegions() != 0 ||
        hasAnyQuantumValue(&op))
      resetValueGrouping();

    for (Region &nestedRegion : op.getRegions())
      for (Block &nestedBlock : nestedRegion)
        scanBlock(nestedBlock);
  }

  flushGroupIfNonEmpty(block, currTextualUnitaryOps);
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::performAnalysis(
    Operation *operation) {
  auto funcOp = dyn_cast<func::FuncOp>(operation);
  if (!funcOp)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Function to analyze: " << funcOp.getName()
                          << '\n');

  for (Region &region : funcOp->getRegions())
    for (Block &block : region)
      scanBlock(block);

  populateGroupBoundaries();

  LLVM_DEBUG(llvm::dbgs() << "Found " << groups.size()
                          << " unitary group(s)\n");
}
