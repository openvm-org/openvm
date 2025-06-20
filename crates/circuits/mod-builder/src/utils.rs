use num_bigint::BigUint;

// Use this when num_limbs is not a constant.
// little endian.
// Warning: This function only returns the last NUM_LIMBS bytes of
//          the input, while the input can have more than that.
#[inline(always)]
pub fn biguint_to_limbs_vec(x: &BigUint, num_limbs: usize) -> Vec<u8> {
    x.to_bytes_le()
        .into_iter()
        .chain(std::iter::repeat(0u8))
        .take(num_limbs)
        .collect()
}
