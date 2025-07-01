use crate::INT256_NUM_LIMBS;

#[inline(always)]
pub(crate) fn u256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
    for i in (0..4).rev() {
        if rs1_u64[i] != rs2_u64[i] {
            return rs1_u64[i] < rs2_u64[i];
        }
    }
    false
}

#[inline(always)]
pub(crate) fn i256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    // true for negative. false forpos
    let rs1_sign = (rs1[INT256_NUM_LIMBS - 1] & 0x80) > 0;
    let rs2_sign = rs2[INT256_NUM_LIMBS - 1] & 0x80 > 0;
    let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
    for i in (0..4).rev() {
        if rs1_u64[i] != rs2_u64[i] {
            return (rs1_u64[i] < rs2_u64[i]) ^ rs1_sign ^ rs2_sign;
        }
    }
    false
}
