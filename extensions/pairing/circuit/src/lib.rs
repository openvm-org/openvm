#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]

pub use openvm_pairing_guest::{
    bls12_381::{BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_ECC_STRUCT_NAME},
    bn254::BN254_COMPLEX_STRUCT_NAME,
};

mod config;
mod fp12;
mod pairing_extension;

pub use config::*;
pub use fp12::*;
pub use pairing_extension::*;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::*;
        pub use cuda::Rv32PairingGpuBuilder as Rv32PairingBuilder;
    } else {
        pub use config::Rv32PairingCpuBuilder as Rv32PairingBuilder;
    }
}
