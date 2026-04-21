use eyre::Result;

use super::utils::{self, ExecutionMode::Pure};

#[test]
fn test_fib() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32im-fib-from-as", utils::DATA), Pure)
}

#[test]
fn test_exp() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32im-exp-from-as", utils::DATA), Pure)
}

#[test]
fn test_terminate() -> Result<()> {
    utils::run_and_compare(&format!("{}/rv32im-terminate-from-as", utils::DATA), Pure)
}

#[test]
fn test_bubblesort() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/bubblesort/elf/openvm-bubblesort-program.elf",
            utils::BENCH
        ),
        Pure,
    )
}

#[test]
fn test_quicksort() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/quicksort/elf/openvm-quicksort-program.elf",
            utils::BENCH
        ),
        Pure,
    )
}

#[test]
fn test_fibonacci_recursive() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/fibonacci_recursive/elf/openvm-fibonacci-recursive-program.elf",
            utils::BENCH
        ),
        Pure,
    )
}

#[test]
fn test_fibonacci_iterative() -> Result<()> {
    utils::run_and_compare(
        &format!(
            "{}/fibonacci_iterative/elf/openvm-fibonacci-iterative-program.elf",
            utils::BENCH
        ),
        Pure,
    )
}

// TODO: Re-enable once the required extension coverage is restored here.
// #[test]
// #[ignore = "slow: large benchmark program"]
// fn test_revm_snailtracer() -> Result<()> {
//     utils::run_and_compare(
//         &format!(
//             "{}/revm_snailtracer/elf/openvm-revm-snailtracer.elf",
//             utils::BENCH
//         ),
//         Pure,
//     )
// }
