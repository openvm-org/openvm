use clap::Parser;
use eyre::Result;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_benchmarks::utils::BenchmarkCli;
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

    let elf = args.build_bench_program("bincode")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        //let file_data = include_bytes!("../../programs/bincode/minecraft_savedata.bin");
        //let stdin = StdIn::from_bytes(file_data);
        let mut stdin = StdIn::default();
        for _ in 0..40000 {
            stdin.write(&vec![33u32; 100])
        }
        let mut config = Rv32ImConfig::default();
        config.system = config.system.with_max_segment_len(600000);
        args.bench_from_exe("bincode", config, exe, stdin)
    })
}
