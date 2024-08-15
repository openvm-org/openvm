use clap::{Parser, Subcommand};
use olap::commands::parse_afo_file;

use crate::{
    commands::{
        benchmark::{benchmark_execute, benchmark_multitier_execute},
        multitier_rw::{run_mtrw_bench, MultitierRwCommand},
        predicate::{run_bench_predicate, PredicateCommand},
        rw::{run_bench_rw, RwCommand},
        vm::{
            vm_fib_program::benchmark_fib_program,
            vm_fib_verifier_program::benchmark_fib_verifier_program,
            vm_verify_fibair::benchmark_verify_fibair, VmCommand,
        },
    },
    config::benchmark_data::{
        benchmark_data_multitier_rw, benchmark_data_predicate, benchmark_data_rw,
    },
    utils::table_gen::{
        generate_incremental_afi_rw, generate_random_afi_rw, generate_random_multitier_afi_rw,
    },
};

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS Benchmark")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(name = "mtrw", about = "Benchmark Multitier Read/Write")]
    /// Read/Write functions
    MtRw(MultitierRwCommand),

    #[command(name = "rw", about = "Benchmark Read/Write")]
    /// Read/Write functions
    Rw(RwCommand),

    #[command(name = "predicate", about = "Benchmark Predicate")]
    /// Predicate functions
    Predicate(PredicateCommand),

    #[command(name = "vm_fib_program", about = "Benchmark VM Fibonacci Program")]
    VmFibProgram(VmCommand),

    #[command(name = "vm_verify_fibair", about = "Benchmark VM Verify FibAir")]
    VmVerifyFibAir(VmCommand),

    #[command(
        name = "vm_fib_verifier_program",
        about = "Benchmark VM Fibonacci Verifier Program"
    )]
    VmFibVerifierProgram(VmCommand),
}

impl Cli {
    pub fn run() {
        let cli = Self::parse();
        match cli.command {
            Commands::MtRw(mtrw) => {
                let benchmark_name = "MultitierReadWrite".to_string();
                let scenario = format!("New Tree: {}", mtrw.new_tree);
                let common = mtrw.common;
                let extra_data = format!("{}", mtrw.new_tree);
                benchmark_multitier_execute(
                    benchmark_name,
                    scenario,
                    common,
                    extra_data,
                    mtrw.start_idx,
                    run_mtrw_bench,
                    benchmark_data_multitier_rw,
                    generate_random_multitier_afi_rw,
                )
                .unwrap();
            }
            Commands::Rw(rw) => {
                let benchmark_name = "ReadWrite".to_string();
                let scenario = format!("r{}%, w{}%", rw.percent_reads, rw.percent_writes);
                let common = rw.common;
                let extra_data = format!("{} {}", rw.percent_reads, rw.percent_writes);
                benchmark_execute(
                    benchmark_name,
                    scenario,
                    common,
                    extra_data,
                    run_bench_rw,
                    benchmark_data_rw,
                    generate_random_afi_rw,
                )
                .unwrap();
            }
            Commands::Predicate(predicate) => {
                let benchmark_name = "Predicate".to_string();
                let afo = parse_afo_file(predicate.afo_file);
                let args = afo.operations[0].args.clone();
                let scenario = format!("{} {}", args[2], args[3]);
                let common = predicate.common;
                let extra_data = format!("0 100 {} {}", args[2], args[3]);
                benchmark_execute(
                    benchmark_name,
                    scenario,
                    common,
                    extra_data,
                    run_bench_predicate,
                    benchmark_data_predicate,
                    generate_incremental_afi_rw,
                )
                .unwrap();
            }
            Commands::VmFibProgram(vm) => benchmark_fib_program(vm.n).unwrap(),
            Commands::VmVerifyFibAir(vm) => benchmark_verify_fibair(vm.n).unwrap(),
            Commands::VmFibVerifierProgram(vm) => benchmark_fib_verifier_program(vm.n).unwrap(),
        }
    }
}
