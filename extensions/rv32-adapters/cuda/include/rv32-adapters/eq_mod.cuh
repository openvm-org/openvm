#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;

template <typename T, size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
struct Rv32IsEqualModAdapterCols {
    ExecutionState<T> from_state;
    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][NUM_LANES];
    T rd_ptr;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> writes_aux;
};

template <size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
struct Rv32IsEqualModAdapterRecord {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_val[NUM_READS];
    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord heap_read_aux[NUM_READS][NUM_LANES];
    uint32_t rd_ptr;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

template <size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
struct Rv32IsEqualModAdapter {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;
    size_t address_bits;

    template <typename T>
    using Cols = Rv32IsEqualModAdapterCols<T, NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>;

    __device__ Rv32IsEqualModAdapter(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        size_t addr_bits,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup),
          address_bits(addr_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        Rv32IsEqualModAdapterRecord<NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS> record
    ) {
        const uint32_t ts = record.timestamp;

        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.timestamp);

        for (size_t i = 0; i < NUM_READS; i++) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptr[i]);

            const uint8_t *val_bytes = reinterpret_cast<const uint8_t *>(&record.rs_val[i]);
            COL_WRITE_ARRAY(row, Cols, rs_val[i], val_bytes);

            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, rs_read_aux[i])),
                record.rs_read_aux[i].prev_timestamp,
                ts + i
            );
        }

        constexpr uint32_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        const uint32_t limb_shift = 1u << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits);
        const uint32_t high_limb_0 = (record.rs_val[0] >> MSL_SHIFT) * limb_shift;
        const uint32_t high_limb_1 =
            (NUM_READS > 1) ? (record.rs_val[1] >> MSL_SHIFT) * limb_shift : 0u;
        bitwise_lookup.add_range(high_limb_0, high_limb_1);

        constexpr size_t TOTAL_READ_SIZE = NUM_LANES * LANE_SIZE;
        assert(TOTAL_READ_SIZE == TOTAL_LIMBS);
        for (size_t i = 0; i < NUM_READS; i++) {
            for (size_t j = 0; j < NUM_LANES; j++) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, heap_read_aux[i][j])),
                    record.heap_read_aux[i][j].prev_timestamp,
                    ts + NUM_READS + i * NUM_LANES + j
                );
            }
        }

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);
        COL_WRITE_ARRAY(row, Cols, writes_aux.prev_data, record.writes_aux.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, writes_aux)),
            record.writes_aux.prev_timestamp,
            ts + NUM_READS + NUM_READS * NUM_LANES
        );
    }
};