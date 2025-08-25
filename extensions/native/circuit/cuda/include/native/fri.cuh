#pragma once

#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"
#include <algorithm>

using namespace native;
using namespace program;

/// Every row starts with these columns.
template <typename T> struct GeneralCols {
    /// Whether the row is a workload row.
    T is_workload_row;
    /// Whether the row is an instruction row.
    T is_ins_row;
    /// For Instruction1 rows, the initial timestamp of the FRI_REDUCED_OPENING instruction.
    /// For Workload rows, the final timestamp after processing the next elements minus
    /// `INSTRUCTION_READS`. For Instruction2 rows, unused.
    T timestamp;
};

template <typename T> struct DataCols {
    /// For Instruction1 rows, `mem[a_ptr_ptr]`.
    /// For Workload rows, the pointer in a-values after increment.
    T a_ptr;
    /// Indicates whether to write a-value or read it.
    /// For Instruction1 rows, `1 - mem[is_init_ptr]`.
    /// For Workload rows, whether we are writing the a-value or reading it; fixed for entire
    /// workload/instruction block.
    T write_a;
    /// For Instruction1 rows, `mem[b_ptr_ptr]`.
    /// For Workload rows, the pointer in b-values after increment.
    T b_ptr;
    /// For Instruction1 rows, the value read from `mem[length_ptr]`.
    /// For Workload rows, the workload row index from the top. *Not* the index into a-values and
    /// b-values. (Note: idx increases within a workload/instruction block, while timestamp, a_ptr,
    /// and b_ptr decrease.)
    T idx;
    /// For both Instruction1 and Workload rows, equal to sum_{k=0}^{idx} alpha^{len-i} (b_i -
    /// a_i). Instruction1 rows constrain this to be the result written to `mem[result_ptr]`.
    T result[EXT_DEG];
    /// The alpha to use in this instruction. Fixed across workload rows; Instruction1 rows read
    /// this from `mem[alpha_ptr]`.
    T alpha[EXT_DEG];
};

/// Prefix of `WorkloadCols` and `Instruction1Cols`
template <typename T> struct PrefixCols {
    GeneralCols<T> general;
    /// WorkloadCols uses this column as the value of `a` read. Instruction1Cols uses this column
    /// as the `is_first` flag must be set to one. Shared with Instruction2Cols `is_first`.
    T a_or_is_first;
    DataCols<T> data;
};

template <typename T> struct WorkloadCols {
    PrefixCols<T> prefix;

    MemoryWriteAuxCols<T, 1> a_aux;
    /// The value of `b` read.
    T b[EXT_DEG];
    MemoryReadAuxCols<T> b_aux;
};

template <typename T> struct Instruction1Cols {
    PrefixCols<T> prefix;

    T pc;

    T a_ptr_ptr;
    MemoryReadAuxCols<T> a_ptr_aux;

    T b_ptr_ptr;
    MemoryReadAuxCols<T> b_ptr_aux;

    /// Extraneous column that is constrained to write_a * a_or_is_first but has no meaningful
    /// effect. It can be removed along with its constraints without impacting correctness.
    T write_a_x_is_first;
};

template <typename T> struct Instruction2Cols {
    GeneralCols<T> general;
    /// Shared with `a_or_is_first` in other column types. Must be 0 for Instruction2Cols.
    T is_first;

    T length_ptr;
    MemoryReadAuxCols<T> length_aux;

    T alpha_ptr;
    MemoryReadAuxCols<T> alpha_aux;

    T result_ptr;
    MemoryWriteAuxCols<T, EXT_DEG> result_aux;

    T hint_id_ptr;

    T is_init_ptr;
    MemoryReadAuxCols<T> is_init_aux;

    /// Extraneous column that is constrained to write_a * a_or_is_first but has no meaningful
    /// effect. It can be removed along with its constraints without impacting correctness.
    T write_a_x_is_first;
};

// Size constants for the three column types
static const size_t WORKLOAD_SIZE = sizeof(WorkloadCols<uint8_t>);
static const size_t INSN1_SIZE = sizeof(Instruction1Cols<uint8_t>);
static const size_t INSN2_SIZE = sizeof(Instruction2Cols<uint8_t>);
static const size_t OVERALL_SIZE = std::max({WORKLOAD_SIZE, INSN1_SIZE, INSN2_SIZE});