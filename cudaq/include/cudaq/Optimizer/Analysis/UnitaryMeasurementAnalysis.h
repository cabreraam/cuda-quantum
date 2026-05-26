/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/TypeID.h"

namespace cudaq::quake::detail {

/// The reusable analysis result for one function.
///
/// This object is intentionally small and factual: it records source-order
/// operations, classifies unitary and measurement operations, and records the
/// simple set of `cc.if` condition values that control each operation. It does
/// not build measurement slices; slicing is a transform-pass consumer of this
/// analysis.
struct UnitaryMeasurementInfo {
  /// All operations in source walk order, including non-quantum operations. A
  /// consumer can use this to find classical producers as well as quantum ops.
  llvm::SmallVector<mlir::Operation *> opOrder;

  /// Quantum gates, excluding `quake.reset`.
  llvm::SmallVector<mlir::Operation *> unitaryOps;

  /// Operations with the `QuantumMeasure` trait.
  llvm::SmallVector<mlir::Operation *> measurementOps;

  /// Per-operation control dependencies discovered while walking nested
  /// `cc.if` regions. Ops not present in this map are unconditional according
  /// to this analysis.
  mlir::DenseMap<mlir::Operation *, llvm::SetVector<mlir::Value>>
      opControlDependencies;
};

using UnitaryMeasurementInfoMap =
    mlir::DenseMap<mlir::Operation *, UnitaryMeasurementInfo>;

/// MLIR analysis object constructed lazily by
/// `getAnalysis<UnitaryMeasurementAnalysis>()` on the current operation.
struct UnitaryMeasurementAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryMeasurementAnalysis)

  explicit UnitaryMeasurementAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  const UnitaryMeasurementInfoMap &getAnalysisInfo() const { return infoMap; }

private:
  void performAnalysis(mlir::Operation *operation);

  UnitaryMeasurementInfoMap infoMap;
};

} // namespace cudaq::quake::detail
