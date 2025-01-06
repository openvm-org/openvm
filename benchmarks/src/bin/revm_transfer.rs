use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::BenchmarkCli;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::StdIn;
use openvm_stark_sdk::{bench::run_with_metric_collection, p3_baby_bear::BabyBear};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

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
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe(
            "revm_100_transfers",
            Keccak256Rv32Config::default(),
            exe,
            StdIn::default(),
        )
    })
}
