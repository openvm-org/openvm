use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::BenchmarkCli;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::StdIn;
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let elf = args.build_bench_program("regex")?;
    let exe = VmExe::from_elf(
        elf.clone(),
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension),
    )?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let data = include_str!("../../programs/regex/regex_email.txt");

        let fe_bytes = data.to_owned().into_bytes();
        args.bench_from_exe::<_, BabyBearPoseidon2Engine>(
            "regex_program",
            Keccak256Rv32Config::default(),
            exe,
            StdIn::from_bytes(&fe_bytes),
        )
    })
}
