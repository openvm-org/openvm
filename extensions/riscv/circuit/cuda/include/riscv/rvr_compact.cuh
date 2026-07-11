#pragma once

#include "riscv/adapters/alu_u16.cuh"
#include "riscv/adapters/branch.cuh"
#include "riscv/adapters/jalr.cuh"
#include "riscv/adapters/rdwrite.cuh"

// M-GPUDEC (G2): on-device decode of rvr compact wire records.
//
// One-derivation-two-consumers rule: the host builds the operand table with the
// same derive helpers its inline assemblers use (rvr_gpu_decode.rs); this
// header is the only other derivation surface and the three-way differential
// (device-decoded vs host-expanded vs CPU) pins it byte-for-byte.

namespace riscv {

// Mirrors the host `DeviceOperandEntry` (rvr_gpu_decode.rs). One entry per
// program slot, indexed by (from_pc - pc_base) / 4. For the alu3/BaseAluU16
// family: a = rd_ptr, b = rs1_ptr, c = rs2 (register pointer or immediate).
struct RvrOperandEntry {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint8_t flags;
    uint8_t local_opcode;
    uint16_t _reserved;
};
static_assert(sizeof(RvrOperandEntry) == 16, "RvrOperandEntry size drift");
static_assert(alignof(RvrOperandEntry) == 4, "RvrOperandEntry align drift");

static constexpr uint8_t RVR_OPERAND_FLAG_RS2_IMM = 1 << 0;
static constexpr uint8_t RVR_OPERAND_FLAG_RS2_IMM_SIGN = 1 << 1;

// The alu3 wire record (44-byte stride), mirroring the C tracer's
// PreflightAlu3Compact / the host reader in log_native.rs. The u64 wire fields
// are split into u32 word pairs: at a 44-byte stride only 4-byte alignment is
// guaranteed, so u64 loads would fault on device.
struct RvrAlu3Compact {
    uint32_t from_pc;                 // 0
    uint32_t from_timestamp;          // 4
    uint32_t reads_prev_timestamp[2]; // 8
    uint32_t write_prev_timestamp;    // 16
    uint32_t write_prev_data[2];      // 20: u64 (lo, hi)
    uint32_t b[2];                    // 28: u64 (lo, hi)
    uint32_t c[2];                    // 36: u64 (lo, hi)
};
static_assert(sizeof(RvrAlu3Compact) == 44, "RvrAlu3Compact stride drift");
static_assert(alignof(RvrAlu3Compact) == 4, "RvrAlu3Compact align drift");

// Little-endian u16 limb `i` of a split u64 wire field.
__device__ __forceinline__ uint16_t rvr_u16_limb(uint32_t const (&words)[2], size_t i) {
    return (uint16_t)(words[i / 2] >> ((i % 2) * 16));
}

// Decode one alu3 wire record + operand-table entry into the full BaseAluU16
// adapter record — the device mirror of the host's
// fill_base_alu_u16_from_compact over derive_base_alu_u16_operands.
__device__ __forceinline__ Rv64BaseAluU16AdapterRecord
rvr_decode_alu3_alu_u16(RvrAlu3Compact const &rec, RvrOperandEntry const &entry) {
    Rv64BaseAluU16AdapterRecord out;
    out.from_pc = rec.from_pc;
    out.from_timestamp = rec.from_timestamp;
    out.rd_ptr = entry.a;
    out.rs1_ptr = entry.b;
    out.rs2 = entry.c;
    // rs2_as: 1 = register read, 0 = immediate (RV64_REGISTER_AS / RV64_IMM_AS).
    out.rs2_as = (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) ? 0 : 1;
    out.rs2_imm_sign = (entry.flags & RVR_OPERAND_FLAG_RS2_IMM_SIGN) ? 1 : 0;
    out.reads_aux[0].prev_timestamp = rec.reads_prev_timestamp[0];
    out.reads_aux[1].prev_timestamp = rec.reads_prev_timestamp[1];
    out.writes_aux.prev_timestamp = rec.write_prev_timestamp;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.writes_aux.prev_data[i] = rvr_u16_limb(rec.write_prev_data, i);
    }
    return out;
}

// Flags for the wr1/rw1 (conditional-write) formats.
static constexpr uint8_t RVR_OPERAND_FLAG_WRITE_ENABLED = 1 << 2;
static constexpr uint8_t RVR_OPERAND_FLAG_IS_JAL = 1 << 3;
static constexpr uint8_t RVR_OPERAND_FLAG_JALR_IMM_SIGN = 1 << 4;

// branch2 wire record (32-byte stride): two register reads, no write.
struct RvrBranch2Compact {
    uint32_t from_pc;                 // 0
    uint32_t from_timestamp;          // 4
    uint32_t reads_prev_timestamp[2]; // 8
    uint32_t b[2];                    // 16: u64 (lo, hi)
    uint32_t c[2];                    // 24: u64 (lo, hi)
};
static_assert(sizeof(RvrBranch2Compact) == 32, "RvrBranch2Compact stride drift");

// wr1 wire record (20-byte stride): one conditional register write.
struct RvrWr1Compact {
    uint32_t from_pc;              // 0
    uint32_t from_timestamp;       // 4
    uint32_t write_prev_timestamp; // 8
    uint32_t write_prev_data[2];   // 12: u64 (lo, hi)
};
static_assert(sizeof(RvrWr1Compact) == 20, "RvrWr1Compact stride drift");

// rw1 wire record (32-byte stride): one read + one conditional write.
struct RvrRw1Compact {
    uint32_t from_pc;              // 0
    uint32_t from_timestamp;       // 4
    uint32_t read_prev_timestamp;  // 8
    uint32_t write_prev_timestamp; // 12
    uint32_t b[2];                 // 16: u64 (lo, hi)
    uint32_t write_prev_data[2];   // 24: u64 (lo, hi)
};
static_assert(sizeof(RvrRw1Compact) == 32, "RvrRw1Compact stride drift");

// Device mirror of the host fill_branch_adapter_from_compact: entry a = rs1_ptr,
// b = rs2_ptr.
__device__ __forceinline__ Rv64BranchAdapterRecord
rvr_decode_branch2_adapter(RvrBranch2Compact const &rec, RvrOperandEntry const &entry) {
    Rv64BranchAdapterRecord out;
    out.from_pc = rec.from_pc;
    out.from_timestamp = rec.from_timestamp;
    out.rs1_ptr = entry.a;
    out.rs2_ptr = entry.b;
    out.reads_aux[0].prev_timestamp = rec.reads_prev_timestamp[0];
    out.reads_aux[1].prev_timestamp = rec.reads_prev_timestamp[1];
    return out;
}

// Device mirror of fill_rdwrite_adapter_from_compact: entry a = rd_ptr; the
// disabled path (suppressed x0 write) uses rd_ptr == UINT32_MAX.
__device__ __forceinline__ Rv64RdWriteAdapterRecord
rvr_decode_wr1_adapter(RvrWr1Compact const &rec, RvrOperandEntry const &entry) {
    Rv64RdWriteAdapterRecord out;
    out.from_pc = rec.from_pc;
    out.from_timestamp = rec.from_timestamp;
    if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
        out.rd_ptr = entry.a;
        out.rd_aux_record.prev_timestamp = rec.write_prev_timestamp;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            out.rd_aux_record.prev_data[i] = rvr_u16_limb(rec.write_prev_data, i);
        }
    } else {
        out.rd_ptr = UINT32_MAX;
        out.rd_aux_record.prev_timestamp = 0;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            out.rd_aux_record.prev_data[i] = 0;
        }
    }
    return out;
}

// Device mirror of the host assemble_jalr_inline adapter fill: entry a = rd_ptr,
// b = rs1_ptr.
__device__ __forceinline__ Rv64JalrAdapterRecord
rvr_decode_rw1_jalr_adapter(RvrRw1Compact const &rec, RvrOperandEntry const &entry) {
    Rv64JalrAdapterRecord out;
    out.from_pc = rec.from_pc;
    out.from_timestamp = rec.from_timestamp;
    out.rs1_ptr = entry.b;
    out.reads_aux.prev_timestamp = rec.read_prev_timestamp;
    if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
        out.rd_ptr = entry.a;
        out.writes_aux.prev_timestamp = rec.write_prev_timestamp;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            out.writes_aux.prev_data[i] = rvr_u16_limb(rec.write_prev_data, i);
        }
    } else {
        out.rd_ptr = UINT32_MAX;
        out.writes_aux.prev_timestamp = 0;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            out.writes_aux.prev_data[i] = 0;
        }
    }
    return out;
}

// Device mirror of the host run_jal_lui_value: JAL -> u16 limbs of (pc + 4)
// (the immediate is unused); LUI -> u16 limbs of (imm << 12) with the high
// limb's sign extended into limbs 2..3.
__device__ __forceinline__ void rvr_jal_lui_rd_data(
    bool is_jal, uint32_t pc, uint32_t imm, uint16_t (&out)[BLOCK_FE_WIDTH]
) {
    if (is_jal) {
        uint32_t rd_low = pc + 4u;
        out[0] = (uint16_t)rd_low;
        out[1] = (uint16_t)(rd_low >> 16);
        out[2] = 0;
        out[3] = 0;
    } else {
        uint32_t rd_low = imm << 12;
        uint16_t hi = (uint16_t)(rd_low >> 16);
        uint16_t sign = (hi >> 15) & 1 ? UINT16_MAX : 0;
        out[0] = (uint16_t)rd_low;
        out[1] = hi;
        out[2] = sign;
        out[3] = sign;
    }
}

// The operand-table entry for a wire record, given the table base and pc base.
__device__ __forceinline__ RvrOperandEntry const &rvr_operand_entry(
    RvrOperandEntry const *table, uint32_t pc_base, uint32_t from_pc
) {
    return table[(from_pc - pc_base) / 4u];
}

} // namespace riscv
