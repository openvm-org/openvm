#[path = "utils.rs"]
mod utils;

use eyre::Result;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use utils::{ExecutionMode::MeteredCost, F};

// ── Prebuilt ELF tests (fast) ───────────────────────────────────────────────

#[test]
fn test_metered_cost_fib() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32im-fib-from-as", utils::DATA), MeteredCost)
}

#[test]
fn test_metered_cost_bubblesort() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/bubblesort/elf/openvm-bubblesort-program.elf",
            utils::BENCH
        ),
        MeteredCost,
    )
}

#[test]
fn test_metered_cost_fibonacci_recursive() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/fibonacci_recursive/elf/openvm-fibonacci-recursive-program.elf",
            utils::BENCH
        ),
        MeteredCost,
    )
}

#[test]
fn test_metered_cost_rv32ui_add() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-add", utils::RVTEST), MeteredCost)
}

#[test]
fn test_metered_cost_rv32ui_sw() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-sw", utils::RVTEST), MeteredCost)
}

#[test]
fn test_metered_cost_rv32ui_lw() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-lw", utils::RVTEST), MeteredCost)
}

#[test]
fn test_metered_cost_rv32ui_jal() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-jal", utils::RVTEST), MeteredCost)
}

#[test]
fn test_metered_cost_rv32um_mul() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32um-p-mul", utils::RVTEST), MeteredCost)
}

// ── Guest program tests (slow: rebuild + keygen) ────────────────────────────

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_hint() -> Result<()> {
    let input = vec![[0u8, 1, 2, 3].map(F::from_u8).to_vec()];
    utils::build_and_compare("hint", &[], input, MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_read() -> Result<()> {
    utils::build_and_compare("read", &[], vec![utils::read_program_input()], MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_reveal() -> Result<()> {
    utils::build_and_compare("reveal", &[], vec![], MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_print() -> Result<()> {
    utils::build_and_compare("print", &[], vec![], MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_getrandom() -> Result<()> {
    utils::build_and_compare("getrandom", &["getrandom"], vec![], MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_guest_fibonacci() -> Result<()> {
    utils::build_and_compare("fibonacci", &[], vec![], MeteredCost)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_cost_guest_collatz() -> Result<()> {
    utils::build_and_compare("collatz", &[], vec![], MeteredCost)
}
