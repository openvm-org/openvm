#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64BaseAluU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2;    // Pointer if rs2 was a read, immediate value (24-bit signed) otherwise.
    T rs2_as; // 1 if rs2 was a read, 0 if an immediate
    T rs2_imm_sign; // Sign bit of the immediate (0 or 1); 0 when rs2_as == RV64_REGISTER_AS.
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluU16AdapterWriteAuxRecord {
    uint32_t prev_timestamp;
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

struct Rv64BaseAluU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2;
    uint8_t rs2_as;
    uint8_t rs2_imm_sign;
    MemoryReadAuxRecord reads_aux[2];
    Rv64BaseAluU16AdapterWriteAuxRecord writes_aux;
};

struct Rv64BaseAluU16Adapter {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    static constexpr uint32_t LIMB_BITS = 16;

    __device__ Rv64BaseAluU16Adapter(
        VariableRangeChecker rc,
        uint32_t timestamp_max_bits
    )
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluU16AdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64BaseAluU16AdapterCols, from_state.timestamp, record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, rs2, record.rs2);
        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, rs2_as, record.rs2_as);
        COL_WRITE_VALUE(row, Rv64BaseAluU16AdapterCols, rs2_imm_sign, record.rs2_imm_sign);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluU16AdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        if (record.rs2_as != 0) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64BaseAluU16AdapterCols, reads_aux[1])),
                record.reads_aux[1].prev_timestamp,
                record.from_timestamp + 1
            );
        } else {
            RowSlice rs2_aux = row.slice_from(COL_INDEX(Rv64BaseAluU16AdapterCols, reads_aux[1]));
#pragma unroll
            for (size_t i = 0; i < sizeof(MemoryReadAuxCols<uint8_t>); i++) {
                rs2_aux.write(i, 0);
            }
            // Mirror AIR range_check on imm low u16 limb.
            range_checker.add_count(record.rs2 & 0xffffu, LIMB_BITS);
        }

        Fp prev[BLOCK_FE_WIDTH];
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            prev[i] = Fp(uint32_t(record.writes_aux.prev_data[i]));
        }
        COL_WRITE_ARRAY(row, Rv64BaseAluU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluU16AdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );
    }
};
