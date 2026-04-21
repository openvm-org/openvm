#![cfg(feature = "rvr")]

#[path = "rvr_openvm/utils.rs"]
mod utils;

#[path = "rvr_openvm/algebra.rs"]
mod algebra;
#[path = "rvr_openvm/asm_and_bench.rs"]
mod asm_and_bench;
#[path = "rvr_openvm/bigint.rs"]
mod bigint;
#[path = "rvr_openvm/debug_info.rs"]
mod debug_info;
#[path = "rvr_openvm/deferral.rs"]
mod deferral;
#[path = "rvr_openvm/ecc.rs"]
mod ecc;
#[path = "rvr_openvm/guest_programs.rs"]
mod guest_programs;
#[path = "rvr_openvm/keccak.rs"]
mod keccak;
#[path = "rvr_openvm/metered.rs"]
mod metered;
#[path = "rvr_openvm/metered_cost.rs"]
mod metered_cost;
#[path = "rvr_openvm/pairing.rs"]
mod pairing;
#[path = "rvr_openvm/riscv_tests.rs"]
mod riscv_tests;
#[path = "rvr_openvm/sha2.rs"]
mod sha2;
