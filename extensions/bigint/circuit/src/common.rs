use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

#[inline(always)]
pub(crate) fn u256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    // Match old algorithm exactly: run_less_than with is_slt=false
    for i in (0..INT256_NUM_LIMBS).rev() {
        if rs1[i] != rs2[i] {
            return rs1[i] < rs2[i];
        }
    }
    false
}

#[inline(always)]
pub(crate) fn i256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    // Match old algorithm exactly: run_less_than with is_slt=true
    let x_sign = rs1[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    let y_sign = rs2[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    for i in (0..INT256_NUM_LIMBS).rev() {
        if rs1[i] != rs2[i] {
            return (rs1[i] < rs2[i]) ^ x_sign ^ y_sign;
        }
    }
    false
}
