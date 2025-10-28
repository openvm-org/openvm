#pragma once

#include <cstddef>

struct Poseidon2DefaultParams {
    static constexpr size_t SBOX_DEGREE = 7;
    static constexpr size_t HALF_FULL_ROUNDS = 4;
    static constexpr size_t PARTIAL_ROUNDS = 13;
};

struct Poseidon2ParamsS1 : Poseidon2DefaultParams {
    static constexpr size_t SBOX_REGS = 1;
};

struct Poseidon2ParamsS0 : Poseidon2DefaultParams {
    static constexpr size_t SBOX_REGS = 0;
};
