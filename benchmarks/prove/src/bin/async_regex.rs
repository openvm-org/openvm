use std::{
    env::var,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use clap::Parser;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_benchmarks_utils::get_programs_dir;
use openvm_sdk::{
    config::{SdkVmBuilder, SdkVmConfig},
    prover::AppProver,
    DefaultStarkEngine, Sdk, StdIn, F,
};
use openvm_stark_sdk::config::setup_tracing;
use tokio::{spawn, task::spawn_blocking, time::sleep};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    setup_tracing();
    let args = BenchmarkCli::parse();
    let mut config = SdkVmConfig::from_toml(include_str!("../../../guest/regex/openvm.toml"))?;
    if let Some(max_height) = args.max_segment_length {
        config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .max_trace_height = max_height;
    }
    if let Some(max_cells) = args.segment_max_cells {
        config.app_vm_config.as_mut().segmentation_limits.max_cells = max_cells;
    }

    let sdk = Sdk::new(config)?;

    let manifest_dir = get_programs_dir().join("regex");
    let elf = sdk.build(Default::default(), manifest_dir, &None, None)?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let data = include_str!("../../../guest/regex/regex_email.txt");
    let fe_bytes = data.to_owned().into_bytes();
    let input = StdIn::<F>::from_bytes(&fe_bytes);

    let (app_pk, _app_vk) = sdk.app_keygen();

    let max_par_jobs: usize = var("MAX_PAR_JOBS").map(|m| m.parse()).unwrap_or(Ok(1))?;
    let num_jobs: usize = var("NUM_JOBS").map(|m| m.parse()).unwrap_or(Ok(10))?;
    let cur_num_jobs = Arc::new(AtomicUsize::new(0));

    let mut tasks = Vec::with_capacity(num_jobs);
    for idx in 0..num_jobs {
        let cur_num_jobs = cur_num_jobs.clone();
        let app_exe = app_exe.clone();
        let app_pk = app_pk.clone();
        let input = input.clone();
        let task = spawn(async move {
            loop {
                let c = cur_num_jobs.fetch_add(1, Ordering::SeqCst);
                if c < max_par_jobs {
                    tracing::info!("Acquired job {}, cur num jobs {}", idx, c + 1);
                    break;
                }
                cur_num_jobs.fetch_sub(1, Ordering::SeqCst);
                sleep(Duration::from_millis(100)).await;
            }
            let res = spawn_blocking(move || -> eyre::Result<()> {
                let mut prover = AppProver::<DefaultStarkEngine, _>::new(
                    SdkVmBuilder,
                    &app_pk.app_vm_pk,
                    app_exe,
                    app_pk.leaf_verifier_program_commit(),
                )?;
                let _proof = prover.prove(input)?;
                Ok(())
            })
            .await?;
            let prev_num = cur_num_jobs.fetch_sub(1, Ordering::SeqCst);
            tracing::info!("Decrement cur_num_jobs {} to {}", prev_num, prev_num - 1);
            res
        });
        tasks.push(task);
    }
    for task in tasks {
        task.await??;
    }

    Ok(())
}
