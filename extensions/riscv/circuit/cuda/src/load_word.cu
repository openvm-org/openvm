#include "riscv/cores/load.cuh"

using LoadWordCore = LoadWidthCore<WORD_ACCESS_WIDTH>;

template <typename T> struct Rv64LoadWordCols {
    Rv64LoadMultiByteAdapterCols<T> adapter;
    LoadWidthCoreCols<T, WORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/load_word.inc.cuh"
