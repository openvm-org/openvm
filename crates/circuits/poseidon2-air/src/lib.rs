//! This is a wrapper around the Plonky3 [p3_poseidon2_air] used only for integration convenience to
//! get around some complications with field-specific generics associated with Poseidon2.
//! Currently it is only intended for use in OpenVM with BabyBear.
//!
//! We do not recommend external use of this crate, and suggest using the [p3_poseidon2_air] crate directly.

pub use openvm_stark_sdk::p3_baby_bear;
pub use p3_poseidon2;
pub use p3_poseidon2_air;
pub use p3_symmetric;

mod poseidon2;
pub use poseidon2::*;
