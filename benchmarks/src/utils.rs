use std::{fs::read, path::PathBuf};

use clap::{command, Parser};
use eyre::Result;
use openvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use openvm_circuit::arch::{instructions::exe::VmExe, DefaultSegmentationStrategy, VmConfig};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_sdk::{
    commit::commit_app_exe,
    config::{
        AggConfig, AggStarkConfig, AppConfig, Halo2Config, DEFAULT_APP_LOG_BLOWUP,
        DEFAULT_INTERNAL_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP, DEFAULT_ROOT_LOG_BLOWUP,
    },
    keygen::{leaf_keygen, AppProvingKey},
    prover::{
        vm::local::VmLocalProver, AppProver, LeafProvingController, DEFAULT_NUM_CHILDREN_LEAF,
    },
    Sdk, StdIn,
};
use openvm_stark_backend::utils::metrics_span;
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    openvm_stark_backend::Chip,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use tempfile::tempdir;
use tracing::info_span;

type F = BabyBear;
type SC = BabyBearPoseidon2Config;

#[derive(Parser, Debug)]
#[command(allow_external_subcommands = true)]
pub struct BenchmarkCli {
    /// Application level log blowup, default set by the benchmark
    #[arg(short = 'p', long, alias = "app_log_blowup")]
    pub app_log_blowup: Option<usize>,

    /// Aggregation (leaf) level log blowup, default set by the benchmark
    #[arg(short = 'g', long, alias = "leaf_log_blowup")]
    pub leaf_log_blowup: Option<usize>,

    /// Internal level log blowup, default set by the benchmark
    #[arg(short, long, alias = "internal_log_blowup")]
    pub internal_log_blowup: Option<usize>,

    /// Root level log blowup, default set by the benchmark
    #[arg(short, long, alias = "root_log_blowup")]
    pub root_log_blowup: Option<usize>,

    #[arg(long)]
    pub halo2_outer_k: Option<usize>,

    #[arg(long)]
    pub halo2_wrapper_k: Option<usize>,

    #[arg(long)]
    pub kzg_params_dir: Option<PathBuf>,

    /// Max segment length for continuations
    #[arg(short, long, alias = "max_segment_length")]
    pub max_segment_length: Option<usize>,

    /// Whether to execute with additional profiling metric collection
    #[arg(long)]
    pub profiling: bool,
}

impl BenchmarkCli {
    pub fn app_config<VC: VmConfig<BabyBear>>(&self, mut app_vm_config: VC) -> AppConfig<VC> {
        let app_log_blowup = self.app_log_blowup.unwrap_or(DEFAULT_APP_LOG_BLOWUP);
        let leaf_log_blowup = self.leaf_log_blowup.unwrap_or(DEFAULT_LEAF_LOG_BLOWUP);

        app_vm_config.system_mut().profiling = self.profiling;
        if let Some(max_segment_length) = self.max_segment_length {
            app_vm_config.system_mut().set_segmentation_strategy(
                DefaultSegmentationStrategy::new_with_max_segment_len(max_segment_length),
            );
        }
        AppConfig {
            app_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                app_log_blowup,
            )
            .into(),
            app_vm_config,
            leaf_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                leaf_log_blowup,
            )
            .into(),
            compiler_options: CompilerOptions {
                enable_cycle_tracker: self.profiling,
                ..Default::default()
            },
        }
    }

    pub fn agg_config(&self) -> AggConfig {
        let leaf_log_blowup = self.leaf_log_blowup.unwrap_or(DEFAULT_LEAF_LOG_BLOWUP);
        let internal_log_blowup = self
            .internal_log_blowup
            .unwrap_or(DEFAULT_INTERNAL_LOG_BLOWUP);
        let root_log_blowup = self.root_log_blowup.unwrap_or(DEFAULT_ROOT_LOG_BLOWUP);

        let [leaf_fri_params, internal_fri_params, root_fri_params] =
            [leaf_log_blowup, internal_log_blowup, root_log_blowup]
                .map(FriParameters::standard_with_100_bits_conjectured_security);

        AggConfig {
            agg_stark_config: AggStarkConfig {
                leaf_fri_params,
                internal_fri_params,
                root_fri_params,
                profiling: self.profiling,
                compiler_options: CompilerOptions {
                    enable_cycle_tracker: self.profiling,
                    ..Default::default()
                },
                root_max_constraint_degree: root_fri_params.max_constraint_degree(),
                ..Default::default()
            },
            halo2_config: Halo2Config {
                verifier_k: self.halo2_outer_k.unwrap_or(23),
                wrapper_k: self.halo2_wrapper_k,
                profiling: self.profiling,
            },
        }
    }

    pub fn build_bench_program(&self, program_name: &str) -> Result<Elf> {
        let profile = if self.profiling {
            "profiling"
        } else {
            "release"
        }
        .to_string();
        build_bench_program(program_name, profile)
    }

    pub fn bench_from_exe<VC>(
        &self,
        bench_name: impl ToString,
        vm_config: VC,
        exe: impl Into<VmExe<F>>,
        input_stream: StdIn,
    ) -> Result<()>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_config = self.app_config(vm_config);
        bench_from_exe(
            bench_name,
            app_config,
            exe,
            input_stream,
            #[cfg(not(feature = "aggregation"))]
            None,
            #[cfg(feature = "aggregation")]
            Some(self.agg_config().agg_stark_config.leaf_vm_config()),
        )
    }
}

fn get_programs_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    dir.push("programs");
    dir
}

pub fn build_bench_program(program_name: &str, profile: impl ToString) -> Result<Elf> {
    let manifest_dir = get_programs_dir().join(program_name);
    let pkg = get_package(manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_target_dir(target_dir.path())
        .with_profile(profile.to_string());
    if let Err(Some(code)) = build_guest_package(&pkg, &guest_opts, None, &None) {
        std::process::exit(code);
    }
    // Assumes the package has a single target binary
    let elf_path = guest_methods(&pkg, &target_dir, &guest_opts.features, &guest_opts.profile)
        .pop()
        .unwrap();
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}

/// 1. Generate proving key from config.
/// 2. Commit to the exe by generating cached trace for program.
/// 3. Executes runtime
/// 4. Generate trace
/// 5. Generate STARK proofs for each segment (segmentation is determined by `config`)
/// 6. Verify STARK proofs.
///
/// Returns the data necessary for proof aggregation.
pub fn bench_from_exe<VC>(
    bench_name: impl ToString,
    app_config: AppConfig<VC>,
    exe: impl Into<VmExe<F>>,
    input_stream: StdIn,
    leaf_vm_config: Option<NativeConfig>,
) -> Result<()>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let bench_name = bench_name.to_string();
    // 1. Generate proving key from config.
    let app_pk = info_span!("keygen", group = &bench_name).in_scope(|| {
        metrics_span("keygen_time_ms", || {
            AppProvingKey::keygen(app_config.clone())
        })
    });
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = info_span!("commit_exe", group = &bench_name).in_scope(|| {
        metrics_span("commit_exe_time_ms", || {
            commit_app_exe(app_config.app_fri_params.fri_params, exe)
        })
    });
    // 3. Executes runtime
    // 4. Generate trace
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    let app_vk = app_pk.get_app_vk();
    let prover = AppProver::new(app_pk.app_vm_pk, committed_exe).with_program_name(bench_name);
    let app_proof = prover.generate_app_proof(input_stream);
    // 6. Verify STARK proofs, including boundary conditions.
    let sdk = Sdk;
    sdk.verify_app_proof(&app_vk, &app_proof)
        .expect("Verification failed");
    if let Some(leaf_vm_config) = leaf_vm_config {
        let leaf_vm_pk = leaf_keygen(app_config.leaf_fri_params.fri_params, leaf_vm_config);
        let leaf_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            leaf_vm_pk,
            app_pk.leaf_committed_exe,
        );
        let leaf_controller = LeafProvingController {
            num_children: DEFAULT_NUM_CHILDREN_LEAF,
        };
        leaf_controller.generate_proof(&leaf_prover, &app_proof);
    }
    Ok(())
}
