#include "riscv/cores/load_sign_extend.cuh"

using LoadSignExtendWordCore = LoadSignExtendWidthCore<WORD_ACCESS_WIDTH>;

template <typename T> struct LoadSignExtendWordCols {
    LoadMultiByteAdapterCols<T> adapter;
    LoadSignExtendWidthCoreCols<T, WORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/load_sign_extend_word.inc.cuh"
