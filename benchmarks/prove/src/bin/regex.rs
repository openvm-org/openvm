use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::StdIn;
use openvm_stark_sdk::{bench::run_with_metric_collection, p3_baby_bear::BabyBear};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let config = Keccak256Rv32Config::default();
    let elf = args.build_bench_program("regex", &config, None)?;
    let exe = VmExe::from_elf(
        elf.clone(),
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension),
    )?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let data = include_str!("../../../guest/regex/regex_email.txt");

        let fe_bytes = data.to_owned().into_bytes();
        args.bench_from_exe("regex_program", config, exe, StdIn::from_bytes(&fe_bytes))
    })
}
