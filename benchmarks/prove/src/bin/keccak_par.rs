use std::sync::Arc;

use clap::Parser;
use openvm_benchmarks_prove::{default_bench_app_params, BenchmarkCli};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{config::AppConfig, prover::verify_app_proof, DefaultStarkEngine, Sdk, StdIn};
use openvm_sdk_config::{SdkVmConfig, TranspilerConfig};
use openvm_stark_sdk::bench::run_with_metric_collection;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE, FromElf};

#[derive(Parser, Debug)]
struct ParallelCli {
    #[clap(flatten)]
    inner: BenchmarkCli,

    #[clap(long, default_value_t = 2)]
    concurrency: usize,
}

fn main() -> eyre::Result<()> {
    let par_args = ParallelCli::parse();
    let mut vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .keccak(Default::default())
        .build()
        .optimize();
    let args = par_args.inner;
    let concurrency = par_args.concurrency;

    let elf = Elf::decode(
        include_bytes!("../../../guest/keccak256_iter/elf/openvm-keccak256-iter-program.elf"),
        MEM_SIZE as u32,
    )?;
    args.apply_config(&mut vm_config);

    let num_keccak_iters: u64 = 1 << 12;
    let mut stdin = StdIn::default();
    stdin.write(&num_keccak_iters);

    let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
    let memory_dims = vm_config.system.config.memory_config.memory_dimensions();
    let app_config = AppConfig::new(vm_config, default_bench_app_params());
    let main_sdk = Sdk::new(app_config.clone(), Default::default())?;
    let (app_pk, app_vk) = main_sdk.app_keygen();
    let app_vk = Arc::new(app_vk);

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let mut handles = vec![];
        for _ in 0..concurrency {
            let app_pk = app_pk.clone();
            let app_vk = app_vk.clone();
            let exe = exe.clone();
            let stdin = stdin.clone();
            let handle = std::thread::spawn(move || -> eyre::Result<_> {
                // Sdk uses OnceLock for internal caching and is not Clone/Sync,
                // so each thread creates its own instance with the shared app_pk.
                let sdk = Sdk::builder()
                    .app_pk(app_pk)
                    .agg_params(Default::default())
                    .default_transpiler()
                    .build()?;
                let mut prover = sdk.app_prover(exe)?;
                let proof = prover.prove(stdin)?;
                let _ = verify_app_proof::<DefaultStarkEngine>(&app_vk.vk, memory_dims, &proof)?;
                Ok(proof)
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok(())
    })
}
