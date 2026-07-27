#include "riscv/cores/load_sign_extend.cuh"

using LoadSignExtendHalfwordCore = LoadSignExtendWidthCore<HALFWORD_ACCESS_WIDTH>;

template <typename T> struct Rv64LoadSignExtendHalfwordCols {
    Rv64LoadMultiByteAdapterCols<T> adapter;
    LoadSignExtendWidthCoreCols<T, HALFWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/load_sign_extend_halfword.inc.cuh"
