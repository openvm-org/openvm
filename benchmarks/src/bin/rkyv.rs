use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, BenchmarkCli};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::StdIn;
use openvm_stark_sdk::{bench::run_with_metric_collection, p3_baby_bear::BabyBear};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();
    let app_config = args.app_config(Rv32ImConfig::default());

    let elf = args.build_bench_program("rkyv")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let file_data = include_bytes!("../../programs/rkyv/minecraft_savedata.bin");
        let stdin = StdIn::from_bytes(file_data);
        bench_from_exe(
            "fibonacci_program",
            app_config,
            exe,
            stdin,
            #[cfg(feature = "aggregation")]
            true,
            #[cfg(not(feature = "aggregation"))]
            false,
        )
    })
}
