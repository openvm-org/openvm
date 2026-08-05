#include "riscv/adapters/store.cuh"
#include "riscv/cores/store.cuh"

using RevealCore = StoreWidthCore<DOUBLEWORD_ACCESS_WIDTH>;

template <typename T> struct RevealCols {
    StoreMultiByteAdapterCols<T> adapter;
    StoreWidthCoreCols<T, DOUBLEWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/reveal.inc.cuh"
