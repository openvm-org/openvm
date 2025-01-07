#![allow(unused_variables)]
#![allow(unused_imports)]

use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use openvm_circuit::arch::{instructions::exe::VmExe, VmConfig};
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_native_circuit::Rv32WithKernelsConfig;
use openvm_native_transpiler::LongFormTranspilerExtension;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::{config::AppConfig, StdIn};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let elf = args.build_bench_program("test_kernel")?;
    let exe = VmExe::from_elf(
        elf.clone(),
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(LongFormTranspilerExtension),
    )?;

    args.bench_from_exe(
        "test_kernel",
        Rv32WithKernelsConfig::default(),
        exe,
        StdIn::default(),
    );

    Ok(())
}
