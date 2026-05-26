/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/UnitaryMeasurementAnalysis.h"
#include "llvm/Support/Debug.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GROUPUNITARIESANDMSMTSANALYSIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "group-unitaries-and-msmts-analysis"

using namespace mlir;

namespace {
static void dumpControlDependencyMap(
    const cudaq::quake::detail::UnitaryMeasurementInfo &info) {
  LLVM_DEBUG(llvm::dbgs() << "Control dependency map:\n");
  if (info.opControlDependencies.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  <empty>\n");
    return;
  }

  // Print in source order rather than DenseMap order so the analysis test is
  // readable and stable.
  for (auto op : info.opOrder) {
    auto iter = info.opControlDependencies.find(op);
    if (iter == info.opControlDependencies.end())
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Controlled op: " << *op << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Control dependencies:\n");
    for (auto value : iter->second)
      LLVM_DEBUG(llvm::dbgs() << "    " << value << "\n");
  }
}

struct GroupUnitariesAndMsmtsAnalysisPass
    : public cudaq::opt::impl::GroupUnitariesAndMsmtsAnalysisBase<
          GroupUnitariesAndMsmtsAnalysisPass> {
  using GroupUnitariesAndMsmtsAnalysisBase::GroupUnitariesAndMsmtsAnalysisBase;

  void runOnOperation() override {
    auto func = getOperation();
    // MLIR constructs/caches this analysis for the current func op on demand.
    const auto &analysis =
        getAnalysis<cudaq::quake::detail::UnitaryMeasurementAnalysis>();
    const auto &analysisInfo = analysis.getAnalysisInfo();
    auto iter = analysisInfo.find(func);
    assert(iter != analysisInfo.end());
    const auto &info = iter->second;

    LLVM_DEBUG(llvm::dbgs() << "\n\nUnitary/measurement analysis for: "
                            << func.getName() << "\n");
    for (auto op : info.unitaryOps)
      LLVM_DEBUG(llvm::dbgs() << "Unitary op: " << *op << "\n");
    for (auto op : info.measurementOps)
      LLVM_DEBUG(llvm::dbgs() << "Measurement op: " << *op << "\n");
    dumpControlDependencyMap(info);
    LLVM_DEBUG(llvm::dbgs() << "\n");

    markAllAnalysesPreserved();
  }
};
} // namespace
