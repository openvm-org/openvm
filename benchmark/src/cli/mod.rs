use clap::{Parser, Subcommand};
use olap::commands::parse_afo_file;

use crate::{
    commands::{
        benchmark_execute,
        predicate::{run_predicate_bench, PredicateCommand},
        rw::{run_rw_bench, RwCommand},
    },
    utils::table_gen::{generate_incremental_afi_rw, generate_random_afi_rw},
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
    #[command(name = "rw", about = "Benchmark Read/Write")]
    /// Read/Write functions
    Rw(RwCommand),

    #[command(name = "predicate", about = "Benchmark Predicate")]
    /// Predicate functions
    Predicate(PredicateCommand),
}

impl Cli {
    pub fn run() {
        let cli = Self::parse();
        match cli.command {
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
                    run_rw_bench,
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
                    run_predicate_bench,
                    generate_incremental_afi_rw,
                )
                .unwrap();
            }
        }
    }
}
