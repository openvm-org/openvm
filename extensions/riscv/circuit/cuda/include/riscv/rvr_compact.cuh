#pragma once

#include "riscv/adapters/alu_u16.cuh"

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

// The operand-table entry for a wire record, given the table base and pc base.
__device__ __forceinline__ RvrOperandEntry const &rvr_operand_entry(
    RvrOperandEntry const *table, uint32_t pc_base, uint32_t from_pc
) {
    return table[(from_pc - pc_base) / 4u];
}

} // namespace riscv
