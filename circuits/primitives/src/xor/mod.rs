mod bus;
/// Xor via preprocessed lookup table. Can only be used if inputs have less than appoximately 10-bits.
mod lookup;

pub use bus::*;
pub use lookup::*;
