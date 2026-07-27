#include "riscv/cores/store.cuh"

using StoreHalfwordCore = StoreWidthCore<HALFWORD_ACCESS_WIDTH>;

template <typename T> struct Rv64StoreHalfwordCols {
    Rv64StoreMultiByteAdapterCols<T> adapter;
    StoreWidthCoreCols<T, HALFWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/store_halfword.inc.cuh"
