use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

#[inline(always)]
pub fn bytes_to_u64_array(bytes: [u8; INT256_NUM_LIMBS]) -> [u64; 4] {
    // SAFETY: [u8; 32] to [u64; 4] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(bytes) }
}

#[inline(always)]
pub fn u64_array_to_bytes(u64_array: [u64; 4]) -> [u8; INT256_NUM_LIMBS] {
    // SAFETY: [u64; 4] to [u8; 32] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(u64_array) }
}

#[inline(always)]
pub fn bytes_to_u32_array(bytes: [u8; INT256_NUM_LIMBS]) -> [u32; 8] {
    // SAFETY: [u8; 32] to [u32; 8] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(bytes) }
}

#[inline(always)]
pub fn u32_array_to_bytes(u32_array: [u32; 8]) -> [u8; INT256_NUM_LIMBS] {
    // SAFETY: [u32; 8] to [u8; 32] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(u32_array) }
}

#[inline(always)]
pub(crate) fn u256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
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
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
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
        common::{
            bytes_to_u32_array, bytes_to_u64_array, i256_lt, u256_lt, u32_array_to_bytes,
            u64_array_to_bytes,
        },
        INT256_NUM_LIMBS,
    };

    #[test]
    fn test_bytes_to_u64_array_round_trip() {
        let original_bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ];

        let u64_array = bytes_to_u64_array(original_bytes);
        let recovered_bytes = u64_array_to_bytes(u64_array);

        assert_eq!(original_bytes, recovered_bytes);
    }

    #[test]
    fn test_bytes_to_u32_array_round_trip() {
        let original_bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ];

        let u32_array = bytes_to_u32_array(original_bytes);
        let recovered_bytes = u32_array_to_bytes(u32_array);

        assert_eq!(original_bytes, recovered_bytes);
    }

    #[test]
    fn test_endianness_preservation() {
        let bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let u64_array = bytes_to_u64_array(bytes);
        let recovered = u64_array_to_bytes(u64_array);
        assert_eq!(bytes, recovered);
    }

    #[test]
    fn test_u256_lt() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u64; 4] = rng.gen();
            let limbs_b: [u64; 4] = rng.gen();
            let a = U256::from_limbs(limbs_a);
            let b = U256::from_limbs(limbs_b);
            let a_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_a);
            let b_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_b);
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
            let a_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_a);
            let b_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_b);
            assert_eq!(i256_lt(a_u8, b_u8), a < b);
        }
    }
}
