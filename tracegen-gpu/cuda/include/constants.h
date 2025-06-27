#pragma once

#include <cstddef>

namespace riscv {
static const size_t RV32_REGISTER_NUM_LIMBS = 4;
static const size_t RV32_CELL_BITS = 8;
static const size_t RV_J_TYPE_IMM_BITS = 21;
} // namespace riscv

namespace program {
static const size_t PC_BITS = 30;
static const size_t DEFAULT_PC_STEP = 4;
} // namespace program
