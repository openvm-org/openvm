pub use builder::*;
pub use collections::*;
pub use fold::*;
pub use instructions::*;
use p3_field::{ExtensionField, PrimeField, TwoAdicField};
pub use poseidon::{DIGEST_SIZE, PERMUTATION_WIDTH};
pub use ptr::*;
pub use symbolic::*;
pub use types::*;
pub use var::*;

mod bits;
mod builder;
mod collections;
mod fold;
mod instructions;
mod poseidon;
mod ptr;
mod symbolic;
mod types;
mod utils;
mod var;

pub trait Config: Clone + Default {
    type N: PrimeField;
    type F: PrimeField + TwoAdicField;
    type EF: ExtensionField<Self::F> + TwoAdicField;
}
