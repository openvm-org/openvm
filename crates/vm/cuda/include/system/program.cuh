#pragma once

template <typename T> struct ProgramExecutionCols {
    T pc;
    T opcode;
    T a;
    T b;
    T c;
    T d;
    T e;
    T f;
    T g;
};

template <typename T> struct ProgramCachedCols {
    T exec_end;
    ProgramExecutionCols<T> exec;
    T exec_start;
};

template <typename T> struct ProgramCols {
    ProgramCachedCols<T> cached;
    T exec_freq;
};
