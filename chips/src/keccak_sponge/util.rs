use tiny_keccak::keccakf;

use super::columns::KECCAK_WIDTH_U16S;

/// Like tiny-keccak's `keccakf`, but deals with `u16` limbs instead of `u64`
/// limbs.
pub(crate) fn keccakf_u16s(state_u16s: &mut [u16; KECCAK_WIDTH_U16S]) {
    let mut state_u64s: [u64; 25] = core::array::from_fn(|i| {
        state_u16s[i * 4..(i + 1) * 4]
            .iter()
            .rev()
            .fold(0, |acc, &x| (acc << 16) | x as u64)
    });
    keccakf(&mut state_u64s);
    *state_u16s = core::array::from_fn(|i| {
        let u64_limb = state_u64s[i / 4];
        let shift = 16 * (i % 4);
        (u64_limb >> shift) as u16
    });
}
