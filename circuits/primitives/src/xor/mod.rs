pub mod bus;
/// Xor via limb decomposition, which interacts with the `lookup::XorLookupChip`
pub mod limbs;
/// Xor via preprocessed lookup table. Can only be used if inputs have less than appoximately 10-bits.
pub mod lookup;
