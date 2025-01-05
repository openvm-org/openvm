use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, BenchmarkCli};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::StdIn;
use openvm_stark_sdk::{bench::run_with_metric_collection, p3_baby_bear::BabyBear};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use tracing::info_span;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();
    let elf = args.build_bench_program("revm_transfer")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let app_config = args.app_config(Keccak256Rv32Config::default());
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        info_span!("revm 100 transfers").in_scope(|| {
            bench_from_exe("revm_transfer", app_config, exe, StdIn::default(), false)
        })?;
        Ok(())
    })
}
