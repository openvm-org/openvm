#include "riscv/cores/store.cuh"

using StoreDoublewordCore = StoreWidthCore<DOUBLEWORD_ACCESS_WIDTH>;

template <typename T> struct Rv64StoreDoublewordCols {
    Rv64StoreMultiByteAdapterCols<T> adapter;
    StoreWidthCoreCols<T, DOUBLEWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/store_doubleword.inc.cuh"
