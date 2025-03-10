pub use builder::*;
pub use collections::*;
pub use instructions::*;
use openvm_stark_backend::p3_field::{ExtensionField, PrimeField, PrimeField32, TwoAdicField};
pub use poseidon::{DIGEST_SIZE, PERMUTATION_WIDTH};
pub use ptr::*;
pub use select::*;
pub use symbolic::*;
pub use types::*;
pub use utils::{LIMB_BITS, NUM_LIMBS};
pub use var::*;

mod bits;
mod builder;
mod collections;
mod fri;
mod instructions;
mod poseidon;
mod ptr;
mod select;
mod symbolic;
mod types;
mod utils;
mod var;
mod verify_batch;

pub trait Config: Clone + Default {
    type N: PrimeField;
    type F: PrimeField32 + TwoAdicField;
    type EF: ExtensionField<Self::F> + TwoAdicField;
}
