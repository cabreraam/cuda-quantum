/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryMeasurementAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
static void dumpControlDependencyMap(
    const cudaq::quake::detail::UnitaryMeasurementInfo &info,
    llvm::raw_ostream &os) {
  os << "Control dependency map:\n";
  if (info.opControlDependencies.empty()) {
    os << "  <empty>\n";
    return;
  }

  // Print in source order rather than DenseMap order so FileCheck expectations
  // are readable and stable.
  for (Operation *op : info.opOrder) {
    auto iter = info.opControlDependencies.find(op);
    if (iter == info.opControlDependencies.end())
      continue;

    os << "Controlled op: " << *op << "\n";
    os << "  Control dependencies:\n";
    for (Value value : iter->second)
      os << "    " << value << "\n";
  }
}

struct TestUnitaryMeasurementAnalysisPass
    : public PassWrapper<TestUnitaryMeasurementAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestUnitaryMeasurementAnalysisPass)

  StringRef getArgument() const final {
    return "test-unitary-measurement-analysis";
  }

  StringRef getDescription() const final {
    return "Print unitary/measurement analysis results.";
  }

  void runOnOperation() override {
    auto func = getOperation();
    const auto &analysis =
        getAnalysis<cudaq::quake::detail::UnitaryMeasurementAnalysis>();
    const auto &analysisInfo = analysis.getAnalysisInfo();
    auto iter = analysisInfo.find(func);
    assert(iter != analysisInfo.end());
    const auto &info = iter->second;

    llvm::raw_ostream &os = llvm::errs();
    os << "Unitary/measurement analysis for: " << func.getName() << "\n";
    for (Operation *op : info.unitaryOps)
      os << "Unitary op: " << *op << "\n";
    for (Operation *op : info.measurementOps)
      os << "Measurement op: " << *op << "\n";
    dumpControlDependencyMap(info, os);

    markAllAnalysesPreserved();
  }
};
} // namespace

namespace cudaq::test {
void registerTestUnitaryMeasurementAnalysisPass() {
  PassRegistration<TestUnitaryMeasurementAnalysisPass>();
}
} // namespace cudaq::test
