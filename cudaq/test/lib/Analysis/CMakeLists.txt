# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_compile_definitions(cudaq-opt PRIVATE CUDAQ_INCLUDE_TESTS)
target_sources(cudaq-opt PRIVATE TestUnitaryMeasurementAnalysis.cpp)
