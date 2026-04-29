#![cfg(feature = "rvr")]

use eyre::Result;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use rvr_openvm_test_utils::{self as utils, ExecutionMode::*, F};

// ── Prebuilt ELF tests (fast) ───────────────────────────────────────────────

#[test]
fn test_metered_fib() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32im-fib-from-as", utils::DATA), Metered)
}

#[test]
fn test_metered_bubblesort() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/bubblesort/elf/openvm-bubblesort-program.elf",
            utils::BENCH
        ),
        Metered,
    )
}

#[test]
fn test_metered_fibonacci_recursive() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/fibonacci_recursive/elf/openvm-fibonacci-recursive-program.elf",
            utils::BENCH
        ),
        Metered,
    )
}

#[test]
fn test_metered_rv32ui_add() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-add", utils::RVTEST), Metered)
}

#[test]
fn test_metered_rv32ui_sw() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32ui-p-sw", utils::RVTEST), Metered)
}

// ── Guest program tests (slow: rebuild + keygen) ────────────────────────────

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_fibonacci() -> Result<()> {
    utils::build_and_compare("fibonacci", &[], vec![], Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_collatz() -> Result<()> {
    utils::build_and_compare("collatz", &[], vec![], Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_hint() -> Result<()> {
    let input = vec![[0u8, 1, 2, 3].map(F::from_u8).to_vec()];
    utils::build_and_compare("hint", &[], input, Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_read() -> Result<()> {
    utils::build_and_compare("read", &[], vec![utils::read_program_input()], Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_reveal() -> Result<()> {
    utils::build_and_compare("reveal", &[], vec![], Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_print() -> Result<()> {
    utils::build_and_compare("print", &[], vec![], Metered)
}

#[test]
#[ignore = "slow: rebuilds guest + keygen, duplicates basic test"]
fn test_metered_guest_getrandom() -> Result<()> {
    utils::build_and_compare("getrandom", &["getrandom"], vec![], Metered)
}

// ── Multi-segment tests ─────────────────────────────────────────────────────

#[test]
fn test_metered_multiseg_fib() -> Result<()> {
    utils::run_and_compare(
        &format!("{}/rv32im-fib-from-as", utils::DATA),
        MeteredMultiseg {
            max_trace_height: 1 << 10,
        },
    )
}

#[test]
#[ignore = "slow: large program with keygen + segmentation"]
fn test_metered_multiseg_bubblesort() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/bubblesort/elf/openvm-bubblesort-program.elf",
            utils::BENCH
        ),
        MeteredMultiseg {
            max_trace_height: 1 << 14,
        },
    )
}

#[test]
#[ignore = "slow: large program with keygen + segmentation"]
fn test_metered_multiseg_fibonacci_recursive() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/fibonacci_recursive/elf/openvm-fibonacci-recursive-program.elf",
            utils::BENCH
        ),
        MeteredMultiseg {
            max_trace_height: 1 << 16,
        },
    )
}

#[test]
#[ignore = "slow: rebuilds guest + keygen + segmentation"]
fn test_metered_multiseg_fibonacci() -> Result<()> {
    utils::build_and_compare(
        "fibonacci",
        &[],
        vec![],
        MeteredMultiseg {
            max_trace_height: 1 << 10,
        },
    )
}

#[test]
#[ignore = "slow: rebuilds guest + keygen + segmentation"]
fn test_metered_multiseg_collatz() -> Result<()> {
    utils::build_and_compare(
        "collatz",
        &[],
        vec![],
        MeteredMultiseg {
            max_trace_height: 1 << 10,
        },
    )
}
