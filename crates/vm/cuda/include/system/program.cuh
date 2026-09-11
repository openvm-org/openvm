#pragma once

template <typename T> struct ProgramExecutionCols {
    T pc_idx;
    T opcode;
    T a;
    T b;
    T c;
    T d;
    T e;
    T f;
    T g;
};

template <typename T> struct ProgramCols {
    ProgramExecutionCols<T> exec;
    T exec_freq;
};
