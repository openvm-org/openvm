use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

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
    // true for negative. false for positive
    let rs1_sign = rs1[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    let rs2_sign = rs2[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
    for i in (0..4).rev() {
        if rs1_u64[i] != rs2_u64[i] {
            return (rs1_u64[i] < rs2_u64[i]) ^ rs1_sign ^ rs2_sign;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use alloy_primitives::{I256, U256};
    use rand::{prelude::StdRng, Rng, SeedableRng};

    use crate::{
        common::{i256_lt, u256_lt},
        INT256_NUM_LIMBS,
    };

    #[test]
    fn test_u256_lt() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u64; 4] = rng.gen();
            let limbs_b: [u64; 4] = rng.gen();
            let a = U256::from_limbs(limbs_a);
            let b = U256::from_limbs(limbs_b);
            let a_u8: [u8; INT256_NUM_LIMBS] = unsafe { std::mem::transmute(limbs_a) };
            let b_u8: [u8; INT256_NUM_LIMBS] = unsafe { std::mem::transmute(limbs_b) };
            assert_eq!(u256_lt(a_u8, b_u8), a < b);
        }
    }
    #[test]
    fn test_i256_lt() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u64; 4] = rng.gen();
            let limbs_b: [u64; 4] = rng.gen();
            let a = I256::from_limbs(limbs_a);
            let b = I256::from_limbs(limbs_b);
            let a_u8: [u8; INT256_NUM_LIMBS] = unsafe { std::mem::transmute(limbs_a) };
            let b_u8: [u8; INT256_NUM_LIMBS] = unsafe { std::mem::transmute(limbs_b) };
            assert_eq!(i256_lt(a_u8, b_u8), a < b);
        }
    }
}
