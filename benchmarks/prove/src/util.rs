use std::path::PathBuf;

use clap::{command, Parser};
use eyre::Result;
use openvm_benchmarks_utils::{build_elf, get_programs_dir};
use openvm_circuit::{
    arch::{
        verify_single, Executor, MeteredExecutor, PreflightExecutor, SystemConfig, VmBuilder,
        VmConfig, VmExecutionConfig,
    },
    utils::{TestRecordArena as RA, TestStarkEngine as Poseidon2Engine},
};
use openvm_native_circuit::{NativeBuilder as DefaultNativeBuilder, NativeConfig};
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_sdk::{
    config::{
        AggregationConfig, AggregationTreeConfig, AppConfig, Halo2Config, TranspilerConfig,
        DEFAULT_APP_LOG_BLOWUP, DEFAULT_HALO2_VERIFIER_K, DEFAULT_INTERNAL_LOG_BLOWUP,
        DEFAULT_LEAF_LOG_BLOWUP, DEFAULT_ROOT_LOG_BLOWUP,
    },
    keygen::_leaf_keygen,
    prover::{verify_app_proof, vm::new_local_prover, LeafProvingController},
    types::ExecutableFormat,
    GenericSdk, StdIn,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::elf::Elf;
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

    /// Max trace height per chip in segment for continuations
    #[arg(long, alias = "max_segment_length")]
    pub max_segment_length: Option<u32>,

    /// Total cells used in all chips in segment for continuations
    #[arg(long)]
    pub segment_max_cells: Option<usize>,

    /// Controls the arity (num_children) of the aggregation tree
    #[command(flatten)]
    pub agg_tree_config: AggregationTreeConfig,

    /// Whether to execute with additional profiling metric collection
    #[arg(long)]
    pub profiling: bool,
}

impl BenchmarkCli {
    pub fn app_config<VC>(&self, mut app_vm_config: VC) -> AppConfig<VC>
    where
        VC: AsMut<SystemConfig>,
    {
        let app_log_blowup = self.app_log_blowup.unwrap_or(DEFAULT_APP_LOG_BLOWUP);
        let leaf_log_blowup = self.leaf_log_blowup.unwrap_or(DEFAULT_LEAF_LOG_BLOWUP);

        app_vm_config.as_mut().profiling = self.profiling;
        app_vm_config.as_mut().max_constraint_degree = (1 << app_log_blowup) + 1;
        if let Some(max_height) = self.max_segment_length {
            app_vm_config.as_mut().segmentation_limits.max_trace_height = max_height;
        }
        if let Some(max_cells) = self.segment_max_cells {
            app_vm_config.as_mut().segmentation_limits.max_cells = max_cells;
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

    pub fn agg_config(&self) -> AggregationConfig {
        let leaf_log_blowup = self.leaf_log_blowup.unwrap_or(DEFAULT_LEAF_LOG_BLOWUP);
        let internal_log_blowup = self
            .internal_log_blowup
            .unwrap_or(DEFAULT_INTERNAL_LOG_BLOWUP);
        let root_log_blowup = self.root_log_blowup.unwrap_or(DEFAULT_ROOT_LOG_BLOWUP);

        let [leaf_fri_params, internal_fri_params, root_fri_params] =
            [leaf_log_blowup, internal_log_blowup, root_log_blowup]
                .map(FriParameters::standard_with_100_bits_conjectured_security);

        AggregationConfig {
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
        }
    }

    pub fn halo2_config(&self) -> Halo2Config {
        Halo2Config {
            verifier_k: self.halo2_outer_k.unwrap_or(DEFAULT_HALO2_VERIFIER_K),
            wrapper_k: self.halo2_wrapper_k,
            profiling: self.profiling,
        }
    }

    pub fn build_bench_program<VC>(
        &self,
        program_name: &str,
        vm_config: &VC,
        init_file_name: Option<&str>,
    ) -> Result<Elf>
    where
        VC: VmConfig<SC>,
    {
        let profile = if self.profiling {
            "profiling"
        } else {
            "release"
        }
        .to_string();
        let manifest_dir = get_programs_dir().join(program_name);
        vm_config.write_to_init_file(&manifest_dir, init_file_name)?;
        build_elf(&manifest_dir, profile)
    }

    pub fn bench_from_exe<VB, VC>(
        &self,
        bench_name: impl ToString,
        vm_config: VC,
        exe: impl Into<ExecutableFormat>,
        input_stream: StdIn,
    ) -> Result<()>
    where
        VB: VmBuilder<Poseidon2Engine, VmConfig = VC, RecordArena = RA> + Clone + Default,
        VC: VmExecutionConfig<F> + VmConfig<SC> + TranspilerConfig<F>,
        <VC as VmExecutionConfig<F>>::Executor:
            Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, RA>,
    {
        let app_config = self.app_config(vm_config);
        bench_from_exe::<Poseidon2Engine, VB, DefaultNativeBuilder>(
            bench_name,
            app_config,
            exe,
            input_stream,
            #[cfg(not(feature = "aggregation"))]
            None,
            #[cfg(feature = "aggregation")]
            Some(self.agg_config().leaf_vm_config()),
        )
    }
}

/// 1. Generate proving key from config.
/// 2. Commit to the exe by generating cached trace for program.
/// 3. Executes runtime
/// 4. Generate trace
/// 5. Generate STARK proofs for each segment (segmentation is determined by `config`)
/// 6. Verify STARK proofs.
///
/// Returns the data necessary for proof aggregation.
pub fn bench_from_exe<E, VB, NativeBuilder>(
    bench_name: impl ToString,
    app_config: AppConfig<VB::VmConfig>,
    exe: impl Into<ExecutableFormat>,
    input_stream: StdIn,
    leaf_vm_config: Option<NativeConfig>,
) -> Result<()>
where
    E: StarkFriEngine<SC = SC>,
    VB: VmBuilder<E> + Clone + Default,
    VB::VmConfig: TranspilerConfig<F>,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone + Default,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    let bench_name = bench_name.to_string();
    let sdk = GenericSdk::<E, VB, NativeBuilder>::new(app_config.clone())?;
    // 1. Generate proving key from config.
    let (app_pk, app_vk) = info_span!("keygen", group = &bench_name).in_scope(|| sdk.app_keygen());
    // 3. Executes runtime
    // 4. Generate trace
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with
    //    timer.
    let mut prover = sdk.app_prover(exe)?.with_program_name(bench_name);
    let app_proof = prover.prove(input_stream)?;
    // 6. Verify STARK proofs, including boundary conditions.
    verify_app_proof(&app_vk, &app_proof)?;
    if let Some(leaf_vm_config) = leaf_vm_config {
        let leaf_vm_pk = _leaf_keygen(app_config.leaf_fri_params.fri_params, leaf_vm_config)?;
        let vk = leaf_vm_pk.vm_pk.get_vk();
        let mut leaf_prover = new_local_prover(
            sdk.native_builder().clone(),
            &leaf_vm_pk,
            app_pk.leaf_committed_exe.exe.clone(),
        )?;
        let leaf_controller = LeafProvingController {
            num_children: AggregationTreeConfig::default().num_children_leaf,
        };
        let leaf_proofs = leaf_controller.generate_proof(&mut leaf_prover, &app_proof)?;
        for proof in leaf_proofs {
            verify_single(&leaf_prover.vm.engine, &vk, &proof)?;
        }
    }
    Ok(())
}
