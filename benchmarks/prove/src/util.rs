use std::path::PathBuf;

use clap::{command, Parser};
use eyre::Result;
use openvm_benchmarks_utils::{build_elf, get_programs_dir};
use openvm_circuit::arch::{
    execution_mode::metered::segment_ctx::SegmentationLimits, instructions::exe::VmExe,
    verify_single, InsExecutorE1, InsExecutorE2, InstructionExecutor, MatrixRecordArena,
    SystemConfig, VmBuilder, VmConfig, VmExecutionConfig,
};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder};
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_sdk::{
    commit::commit_app_exe,
    config::{
        AggConfig, AggStarkConfig, AggregationTreeConfig, AppConfig, Halo2Config,
        DEFAULT_APP_LOG_BLOWUP, DEFAULT_INTERNAL_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP,
        DEFAULT_ROOT_LOG_BLOWUP,
    },
    keygen::{leaf_keygen, AppProvingKey},
    prover::{vm::new_local_prover, AppProver, LeafProvingController},
    GenericSdk, StdIn,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
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

    /// Max segment length for continuations
    #[arg(short, long, alias = "max_segment_length")]
    pub max_segment_length: Option<usize>,

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
        if let Some(max_segment_length) = self.max_segment_length {
            app_vm_config.as_mut().set_segmentation_limits(
                SegmentationLimits::default().with_max_trace_height(max_segment_length as u32),
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
        app_vm_builder: VB,
        vm_config: VC,
        exe: impl Into<VmExe<F>>,
        input_stream: StdIn,
    ) -> Result<()>
    where
        VB: VmBuilder<BabyBearPoseidon2Engine, VmConfig = VC, RecordArena = MatrixRecordArena<F>>,
        VC: VmExecutionConfig<F> + VmConfig<SC>,
        <VC as VmExecutionConfig<F>>::Executor:
            InsExecutorE1<F> + InsExecutorE2<F> + InstructionExecutor<F>,
    {
        let app_config = self.app_config(vm_config);
        bench_from_exe::<BabyBearPoseidon2Engine, _, NativeCpuBuilder>(
            bench_name,
            app_vm_builder,
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
    app_vm_builder: VB,
    app_config: AppConfig<VB::VmConfig>,
    exe: impl Into<VmExe<F>>,
    input_stream: StdIn,
    leaf_vm_config: Option<NativeConfig>,
) -> Result<()>
where
    E: StarkFriEngine<SC = SC>,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        InsExecutorE1<F> + InsExecutorE2<F> + InstructionExecutor<F, VB::RecordArena>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone + Default,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        InstructionExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    let bench_name = bench_name.to_string();
    // 1. Generate proving key from config.
    let app_pk = info_span!("keygen", group = &bench_name)
        .in_scope(|| AppProvingKey::keygen(app_config.clone()))?;
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = info_span!("commit_exe", group = &bench_name)
        .in_scope(|| commit_app_exe(app_config.app_fri_params.fri_params, exe));
    // 3. Executes runtime
    // 4. Generate trace
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with
    //    timer.
    let app_vk = app_pk.get_app_vk();
    let mut prover = AppProver::<E, _>::new(app_vm_builder, app_pk.app_vm_pk, committed_exe)?
        .with_program_name(bench_name);
    let app_proof = prover.generate_app_proof(input_stream)?;
    // 6. Verify STARK proofs, including boundary conditions.
    let sdk = GenericSdk::<E, NativeBuilder>::new();
    sdk.verify_app_proof(&app_vk, &app_proof)?;
    if let Some(leaf_vm_config) = leaf_vm_config {
        let leaf_vm_pk = leaf_keygen(app_config.leaf_fri_params.fri_params, leaf_vm_config)?;
        let vk = leaf_vm_pk.vm_pk.get_vk();
        let mut leaf_prover = new_local_prover(
            sdk.native_builder().clone(),
            &leaf_vm_pk,
            &app_pk.leaf_committed_exe,
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
