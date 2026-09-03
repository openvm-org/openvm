#include "riscv/cores/load.cuh"

using LoadDoublewordCore = LoadWidthCore<DOUBLEWORD_ACCESS_WIDTH>;

template <typename T> struct LoadDoublewordCols {
    LoadMultiByteAdapterCols<T> adapter;
    LoadWidthCoreCols<T, DOUBLEWORD_ACCESS_WIDTH> core;
};

#include "../rvr/src/load_doubleword.inc.cuh"
