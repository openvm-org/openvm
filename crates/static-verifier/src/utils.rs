#[inline]
pub(crate) fn bits_for_u64(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).max(1)
}

#[inline]
pub(crate) fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).expect("usize value does not fit in u64")
}
