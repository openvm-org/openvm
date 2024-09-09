pub use builder::*;
pub use collections::*;
pub use instructions::*;
pub use modular_arithmetic::*;
use p3_field::{ExtensionField, PrimeField, TwoAdicField};
pub use poseidon::{DIGEST_SIZE, PERMUTATION_WIDTH};
pub use ptr::*;
pub use ref_ptr::*;
pub use select::*;
pub use symbolic::*;
pub use types::*;
pub use var::*;

mod bits;
mod builder;
mod collections;
mod elliptic_curve;
mod instructions;
mod keccak;
mod modular_arithmetic;
mod poseidon;
mod ptr;
mod ref_ptr;
mod select;
mod symbolic;
mod types;
mod utils;
mod var;

pub trait Config: Clone + Default {
    type N: PrimeField;
    type F: PrimeField + TwoAdicField;
    type EF: ExtensionField<Self::F> + TwoAdicField;
}
