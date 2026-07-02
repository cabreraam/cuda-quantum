/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class Operation;
class Block;
class Region;
} // namespace mlir

namespace cudaq::quake::detail {
/// Description of a measurement that consumes one or more output wires from a
/// unitary group.
///
/// Readout boundaries are not themselves unitary operations. They record where
/// the quantum dataflow produced by a group is read out and which value
/// semantics wires continue after that measurement.
struct ReadoutBoundary {
  /// The Quake measurement operation at the boundary.
  cudaq::quake::MeasurementInterface measurement;

  /// Wire values consumed by the measurement.
  mlir::SmallVector<mlir::Value> measuredWires;

  /// Wire values produced by the measurement, one per measured wire.
  mlir::SmallVector<mlir::Value> postMeasurementWires;
};

/// A maximal group of unitary Quake operations discovered by the analysis.
///
/// For reference-semantics IR, the group is a contiguous textual run of unitary
/// operations in a single block. For clean value-semantics wire IR, the group
/// is formed by SSA wire dataflow within one block, so independent wires may
/// remain in the same group across intervening measurements.
struct UnitaryOpGroup {
  UnitaryOpGroup() = default;

  /// Block containing all operations in this group.
  mlir::Block *block = nullptr;

  /// Unitary operations that belong to this group, in analysis discovery order.
  mlir::SmallVector<mlir::Operation *> ops;

  /// Quantum values consumed by this group that are defined outside the group.
  mlir::SmallVector<mlir::Value> quantumInputs;

  /// Non-quantum values consumed by this group that are defined outside the
  /// group.
  mlir::SmallVector<mlir::Value> classicalInputs;

  /// Quantum wire results that leave this group or have no users.
  mlir::SmallVector<mlir::Value> quantumOutputs;

  /// Measurements that consume quantum outputs from this group.
  mlir::SmallVector<ReadoutBoundary> readouts;
};

using UnitaryOpGroups =
    mlir::SmallVector<cudaq::quake::detail::UnitaryOpGroup, 4>;

/// Analysis to group unitary operations within blocks.
///
/// Reference-semantics operations are grouped as maximal contiguous runs of
/// unitary quantum operations. Clean value-semantics operations are grouped by
/// SSA wire dataflow, so measurements only cut the group along the wire or
/// wires they measure. Measurements that consume group outputs are recorded as
/// readout boundaries rather than being added to a group.
struct UnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryOpGroupingAnalysis)

  /// Construct the analysis for a function operation.
  ///
  /// If \p op is not a `func.func`, the analysis leaves the group collection
  /// empty.
  explicit UnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  /// Return all unitary operation groups discovered by the analysis.
  ///
  /// The groups are ordered by the recursive block scan order used by the
  /// analysis.
  const UnitaryOpGroups &getGroups() const { return groups; }

  /// Return the block that contains \p group.
  const mlir::Block *getBlockForGroup(const UnitaryOpGroup &group) const;

  /// Return the group containing \p op, or `nullptr` if \p op is not grouped.
  ///
  /// Non-unitary operations, null operations, and operations from outside the
  /// analyzed function return `nullptr`.
  const UnitaryOpGroup *getGroupContainingOp(mlir::Operation *op) const;

  /// Return the group that produces \p value, or `nullptr` if \p value is not a
  /// quantum output of a discovered group.
  const UnitaryOpGroup *getGroupProducingValue(mlir::Value value) const;

  /// Return the unitary groups contained in \p block.
  ///
  /// Returns an empty vector when \p block is null or no groups were found in
  /// the block.
  mlir::SmallVector<const UnitaryOpGroup *>
  getGroupsIn(const mlir::Block *block) const;

  /// Return true if both operations are in the same unitary group.
  ///
  /// Returns false if either operation is null, non-unitary, or not part of a
  /// discovered unitary group.
  bool inSameGroup(mlir::Operation *op1, mlir::Operation *op2) const;

private:
  /// Compute all unitary groups for \p operation when it is a function.
  void performAnalysis(mlir::Operation *operation);

  /// Scan one block and append any discovered groups to \p groups.
  void scanBlock(mlir::Block &block);

  /// Create an empty group for \p block and update block lookup tables.
  unsigned createGroup(mlir::Block &block);

  /// Add \p op to an existing group and update operation/value lookup tables.
  void addOpToGroup(unsigned groupIndex, mlir::Operation *op);

  /// Populate input/output/readout metadata after all groups are discovered.
  void populateGroupBoundaries();

  /// Materialize a pending contiguous textual group, if one exists.
  void flushGroupIfNonEmpty(
      mlir::Block &block,
      mlir::SmallVectorImpl<mlir::Operation *> &currUnitaryOps);

  /// Discovered unitary groups.
  UnitaryOpGroups groups;

  /// Map from grouped unitary operation to its group index.
  llvm::DenseMap<mlir::Operation *, unsigned> opToGroupIndex;

  /// Map from value-semantics wire result to the group that produced it.
  llvm::DenseMap<mlir::Value, unsigned> valueToGroupIndex;

  /// Map from block to the group indices discovered inside that block.
  llvm::DenseMap<const mlir::Block *, mlir::SmallVector<unsigned>>
      blockToGroupIndices;
};
} // namespace cudaq::quake::detail
