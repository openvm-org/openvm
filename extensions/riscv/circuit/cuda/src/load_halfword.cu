#include "riscv/cores/load.cuh"

using LoadHalfwordCore = LoadWidthCore<HALFWORD_ACCESS_WIDTH>;

template <typename T> struct Rv64LoadHalfwordCols {
    Rv64LoadMultiByteAdapterCols<T> adapter;
    LoadWidthCoreCols<T, HALFWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/load_halfword.inc.cuh"
