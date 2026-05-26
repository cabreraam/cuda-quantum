/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryMeasurementAnalysis.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {
/// For this analysis, `quake.reset` is not treated as a unitary. It is kept in
/// `opOrder`, though, because consumers may use it as a quantum-state boundary.
static bool isUnitary(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() && !isa<cudaq::quake::ResetOp>(op);
}

static bool isMeasurement(Operation *op) {
  return op->hasTrait<cudaq::QuantumMeasure>();
}

/// Record one operation in source order and classify it if it is a quantum gate
/// or measurement. `activeControlDependencies` is the stack of enclosing
/// `cc.if` conditions at this point in the walk.
static void recordOperation(cudaq::quake::detail::UnitaryMeasurementInfo &info,
                            Operation *op,
                            ArrayRef<Value> activeControlDependencies) {
  info.opOrder.push_back(op);

  if (!activeControlDependencies.empty()) {
    auto &controls = info.opControlDependencies[op];
    for (auto value : activeControlDependencies)
      controls.insert(value);
  }

  if (isUnitary(op))
    info.unitaryOps.push_back(op);
  else if (isMeasurement(op))
    info.measurementOps.push_back(op);
}

static void
collectOperations(Region &region,
                  cudaq::quake::detail::UnitaryMeasurementInfo &info,
                  SmallVectorImpl<Value> &activeControlDependencies);

static void
collectOperations(Block &block,
                  cudaq::quake::detail::UnitaryMeasurementInfo &info,
                  SmallVectorImpl<Value> &activeControlDependencies) {
  for (auto &nestedOp : block) {
    Operation *op = &nestedOp;
    recordOperation(info, op, activeControlDependencies);

    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
      // Keep control-flow handling deliberately simple: every op nested in
      // either branch is marked as dependent on the condition value. We do not
      // record branch polarity or build Boolean path formulas.
      activeControlDependencies.push_back(ifOp.getCondition());
      collectOperations(ifOp.getThenRegion(), info, activeControlDependencies);
      collectOperations(ifOp.getElseRegion(), info, activeControlDependencies);
      activeControlDependencies.pop_back();
      continue;
    }

    for (auto &region : op->getRegions())
      collectOperations(region, info, activeControlDependencies);
  }
}

static void
collectOperations(Region &region,
                  cudaq::quake::detail::UnitaryMeasurementInfo &info,
                  SmallVectorImpl<Value> &activeControlDependencies) {
  for (auto &block : region)
    collectOperations(block, info, activeControlDependencies);
}
} // namespace

void cudaq::quake::detail::UnitaryMeasurementAnalysis::performAnalysis(
    Operation *operation) {
  auto func = dyn_cast<func::FuncOp>(operation);
  if (!func)
    return;

  UnitaryMeasurementInfo info;
  // This vector behaves like a stack while recursively walking nested regions.
  SmallVector<Value> activeControlDependencies;
  collectOperations(func.getBody(), info, activeControlDependencies);
  infoMap.insert({operation, std::move(info)});
}
