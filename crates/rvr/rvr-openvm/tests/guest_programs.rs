#[path = "utils.rs"]
mod utils;

use eyre::Result;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use utils::{ExecutionMode::Pure, F};

#[test]
fn test_guest_fibonacci() -> Result<()> {
    utils::build_and_compare("fibonacci", &[], vec![], Pure)
}

#[test]
fn test_guest_collatz() -> Result<()> {
    utils::build_and_compare("collatz", &[], vec![], Pure)
}

#[test]
fn test_guest_hint() -> Result<()> {
    let input = vec![[0u8, 1, 2, 3].map(F::from_u8).to_vec()];
    utils::build_and_compare("hint", &[], input, Pure)
}

#[test]
fn test_guest_print() -> Result<()> {
    utils::build_and_compare("print", &[], vec![], Pure)
}

#[test]
fn test_guest_reveal() -> Result<()> {
    utils::build_and_compare("reveal", &[], vec![], Pure)
}

#[test]
fn test_guest_read() -> Result<()> {
    utils::build_and_compare("read", &[], vec![utils::read_program_input()], Pure)
}

#[test]
fn test_guest_tiny_mem_test() -> Result<()> {
    utils::build_and_compare("tiny-mem-test", &["heap-embedded-alloc"], vec![], Pure)
}

#[test]
fn test_guest_getrandom() -> Result<()> {
    utils::build_and_compare("getrandom", &["getrandom"], vec![], Pure)
}

#[test]
fn test_guest_getrandom_v02() -> Result<()> {
    utils::build_and_compare("getrandom_v02", &["getrandom-v02/custom"], vec![], Pure)
}
