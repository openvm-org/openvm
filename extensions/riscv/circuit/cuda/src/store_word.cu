#include "riscv/cores/store.cuh"

using StoreWordCore = StoreWidthCore<WORD_ACCESS_WIDTH>;

template <typename T> struct Rv64StoreWordCols {
    Rv64StoreMultiByteAdapterCols<T> adapter;
    StoreWidthCoreCols<T, WORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/store_word.inc.cuh"
