use clap::{command, Parser};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{
    config::{AggregationSystemParams, AppConfig},
    prover::verify_app_proof,
    DefaultStarkEngine, Sdk, StdIn,
};
use openvm_sdk_config::{SdkVmConfig, TranspilerConfig};
use openvm_stark_backend::SystemParams;
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{
        app_params_with_100_bits_security, internal_params_with_100_bits_security,
        leaf_params_with_100_bits_security,
    },
};
use openvm_transpiler::{elf::Elf, FromElf};
use openvm_verify_stark_host::{verify_vm_stark_proof_decoded, vk::NonRootStarkVerifyingKey};

pub const DEFAULT_MAX_SEGMENT: u32 = 1 << 20;
pub const DEFAULT_LOG_STACKED_HEIGHT: usize = 21;

#[derive(Parser, Debug)]
#[command(allow_external_subcommands = true)]
pub struct BenchmarkCli {
    /// Max trace height per chip in segment for continuations
    #[arg(long, alias = "max_segment_length")]
    pub max_segment_length: Option<u32>,

    /// Only runs the app proof
    #[arg(long)]
    pub app_only: bool,

    /// Whether to execute with additional profiling metric collection
    #[arg(long)]
    pub profiling: bool,
    /// Directory containing KZG trusted setup files (for e2e halo2 proving)
    #[arg(long)]
    pub kzg_params_dir: Option<std::path::PathBuf>,
    // #[arg(long)]
    // pub halo2_outer_k: Option<usize>,

    // #[arg(long)]
    // pub halo2_wrapper_k: Option<usize>,
}

impl BenchmarkCli {
    /// Applies CLI-specified segmentation config to the VM config.
    /// The max trace height is always rounded up to the next power of two.
    pub fn apply_config(&self, vm_config: &mut SdkVmConfig) {
        let max_height = self
            .max_segment_length
            .unwrap_or(DEFAULT_MAX_SEGMENT)
            .next_power_of_two();
        vm_config
            .as_mut()
            .segmentation_config
            .limits
            .set_max_trace_height(max_height);
        vm_config.as_mut().profiling = self.profiling;
    }

    pub fn run(&self, mut vm_config: SdkVmConfig, elf: Elf, stdin: StdIn) -> eyre::Result<()> {
        self.apply_config(&mut vm_config);
        if self.app_only {
            run_default_app_benchmark(vm_config, elf, stdin)
        } else {
            run_default_benchmark(vm_config, elf, stdin)
        }
    }
}

pub fn run_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
    leaf_params: SystemParams,
    internal_params: SystemParams,
) -> eyre::Result<()> {
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, app_params);
        let agg_params = AggregationSystemParams {
            leaf: leaf_params,
            internal: internal_params,
        };
        let sdk = Sdk::new(app_config, agg_params)?;
        let (proof, baseline) = sdk.prove(exe, stdin, &[])?;
        #[cfg(feature = "metrics")]
        {
            use openvm_stark_backend::codec::Encode;
            let encoded = proof.encode_to_vec()?;
            let compressed = zstd::encode_all(&encoded[..], 19)?;
            tracing::info!(
                "Proof Size (bytes): {}, Compressed Size: {}",
                encoded.len(),
                compressed.len()
            );
            metrics::gauge!("proof_size_bytes.total").set(encoded.len() as f64);
            metrics::gauge!("proof_size_bytes.compressed").set(compressed.len() as f64);
        }
        let vk = NonRootStarkVerifyingKey {
            mvk: (*sdk.agg_vk()).clone(),
            baseline,
        };
        verify_vm_stark_proof_decoded(&vk, &proof)?;
        Ok(())
    })
}

pub fn run_app_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
) -> eyre::Result<()> {
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let memory_dims = vm_config.system.config.memory_config.memory_dimensions();
        let app_config = AppConfig::new(vm_config, app_params);
        let sdk = Sdk::new(app_config, Default::default())?;
        let (_, app_vk) = sdk.app_keygen();
        let mut prover = sdk.app_prover(exe)?;
        let proof = prover.prove(stdin)?;
        let _ = verify_app_proof::<DefaultStarkEngine>(&app_vk.vk, memory_dims, &proof)?;
        Ok(())
    })
}

pub fn run_default_benchmark(vm_config: SdkVmConfig, elf: Elf, stdin: StdIn) -> eyre::Result<()> {
    run_benchmark(
        vm_config,
        elf,
        stdin,
        default_bench_app_params(),
        leaf_params_with_100_bits_security(),
        internal_params_with_100_bits_security(),
    )
}

pub fn run_default_app_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
) -> eyre::Result<()> {
    run_app_benchmark(vm_config, elf, stdin, default_bench_app_params())
}

pub fn default_bench_app_params() -> SystemParams {
    app_params_with_100_bits_security(DEFAULT_LOG_STACKED_HEIGHT)
}
