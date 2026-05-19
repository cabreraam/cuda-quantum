/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "llvm/Support/Debug.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GROUPUNITARIESANDMSMTSANALYSIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "group-unitaries-and-msmts-analysis"

using namespace mlir;

namespace {
struct Analysis {
  Analysis() = default;

  explicit Analysis(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (op->hasTrait<cudaq::QuantumGate>() &&
          !isa<cudaq::quake::ResetOp>(op)) {
        unitaryOps.push_back(op);
      } else if (op->hasTrait<cudaq::QuantumMeasure>()) {
        measurementOps.push_back(op);
      }
      return WalkResult::advance();
    });
  }

  SmallVector<Operation *> unitaryOps;
  SmallVector<Operation *> measurementOps;
};

struct GroupUnitariesAndMsmtsAnalysis
    : public cudaq::opt::impl::GroupUnitariesAndMsmtsAnalysisBase<
          GroupUnitariesAndMsmtsAnalysis> {
  using GroupUnitariesAndMsmtsAnalysisBase::GroupUnitariesAndMsmtsAnalysisBase;

  void runOnOperation() override {
    auto func = getOperation();
    Analysis analysis(func);

    LLVM_DEBUG(llvm::dbgs() << "After logical grouping analysis:\n" << *func);
    for (auto op : analysis.unitaryOps) {
      LLVM_DEBUG(llvm::dbgs() << "Unitary op: " << *op << "\n");
    }
    for (auto op : analysis.measurementOps) {
      LLVM_DEBUG(llvm::dbgs() << "Measurement op: " << *op << "\n");
    }
  }
};
} // namespace
