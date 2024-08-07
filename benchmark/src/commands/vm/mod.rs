use clap::Parser;

use super::CommonCommands;

mod benchmark_helpers;
pub mod vm_fib_program;
pub mod vm_fib_verifier_program;
pub mod vm_verify_fibair;

pub struct VmBenchmarkConfig {
    pub n: usize,
}

#[derive(Debug, Parser)]
pub struct VmCommand {
    #[arg(
        long = "n-value",
        short = 'n',
        help = "The value of n such that we are computing the n-th Fibonacci number",
        default_value = "2",
        required = true
    )]
    /// The value of n such that we are computing the n-th Fibonacci number
    pub n: usize,

    #[command(flatten)]
    pub common: CommonCommands,
}
