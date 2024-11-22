use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{
        config::{StarkGenericConfig, Val},
        prover::types::Proof,
    },
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_benchmarks::utils::{build_bench_program, BenchmarkCli};
use axvm_circuit::{
    arch::{
        instructions::{exe::AxVmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES},
        SingleSegmentVmExecutor, VmConfig, VmExecutor,
    },
    prover::{local::VmLocalProver, ContinuationVmProver, SingleSegmentVmProver},
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::hints::Hintable;
use axvm_sdk::{
    config::AxVmSdkConfig,
    keygen::AxVmSdkProvingKey,
    prover::RootVerifierLocalProver,
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
};
use axvm_transpiler::axvm_platform::bincode;
use clap::Parser;
use eyre::Result;
use metrics::counter;
use p3_field::{AbstractField, PrimeField32};
use tracing::info_span;

type SC = BabyBearPoseidon2Config;
type F = BabyBear;
const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;
const NUM_CHILDREN_LEAF: usize = 2;
const NUM_CHILDREN_INTERNAL: usize = 2;

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);
    let root_log_blowup = cli_args.root_log_blowup.unwrap_or(2);
    let internal_log_blowup = cli_args.internal_log_blowup.unwrap_or(2);

    let num_segments = 8;
    // Must be larger than RangeTupleCheckerAir.height == 524288
    let segment_len = 1_000_000;
    let axvm_sdk_pk = {
        let axvm_sdk_config = AxVmSdkConfig {
            max_num_user_public_values: NUM_PUBLIC_VALUES,
            app_fri_params: standard_fri_params_with_100_bits_conjectured_security(app_log_blowup),
            leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(agg_log_blowup),
            internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(
                internal_log_blowup,
            ),
            root_fri_params: standard_fri_params_with_100_bits_conjectured_security(
                root_log_blowup,
            ),
            app_vm_config: VmConfig::rv32im()
                .with_num_public_values(NUM_PUBLIC_VALUES)
                .with_max_segment_len(segment_len),
            compiler_options: CompilerOptions {
                enable_cycle_tracker: true,
                ..Default::default()
            },
        };
        AxVmSdkProvingKey::keygen(axvm_sdk_config)
    };

    let app_committed_exe = generate_fib_exe(axvm_sdk_pk.app_vm_pk.fri_params);

    let n = 800_000u64;
    let app_input: Vec<_> = bincode::serde::encode_to_vec(n, bincode::config::standard())?
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    run_with_metric_collection("OUTPUT_PATH", || {
        let app_proofs = info_span!(
            "Fibonacci Continuation Program",
            group = "fibonacci_continuation_program"
        )
        .in_scope(|| {
            let mut vm_config = axvm_sdk_pk.app_vm_pk.vm_config.clone();
            vm_config.collect_metrics = true;
            let vm = VmExecutor::new(vm_config);
            let execution_results = vm
                .execute_segments(app_committed_exe.exe.clone(), vec![app_input.clone()])
                .unwrap();
            assert_eq!(execution_results.len(), num_segments);
            let app_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                axvm_sdk_pk.app_vm_pk.clone(),
                app_committed_exe.clone(),
            );
            counter!("fri.log_blowup").absolute(axvm_sdk_pk.app_vm_pk.fri_params.log_blowup as u64);
            ContinuationVmProver::prove(&app_prover, vec![app_input])
        });

        let leaf_proofs = info_span!("leaf verifier", group = "leaf_verifier").in_scope(|| {
            let leaf_inputs =
                LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proofs, NUM_CHILDREN_LEAF);
            let leaf_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                axvm_sdk_pk.leaf_vm_pk.clone(),
                axvm_sdk_pk.leaf_committed_exe.clone(),
            );
            leaf_inputs
                .into_iter()
                .enumerate()
                .map(|(leaf_idx, input)| {
                    info_span!("leaf verifier proof", index = leaf_idx).in_scope(|| {
                        single_segment_execute_and_prove(&leaf_prover, input.write_to_stream())
                    })
                })
                .collect::<Vec<_>>()
        });
        let final_internal_proof = {
            let internal_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                axvm_sdk_pk.internal_vm_pk,
                axvm_sdk_pk.internal_committed_exe.clone(),
            );
            let mut internal_node_idx = 0;
            let mut internal_node_height = 0;
            let mut proofs = leaf_proofs;
            while proofs.len() > 1 {
                let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                    axvm_sdk_pk
                        .internal_committed_exe
                        .committed_program
                        .prover_data
                        .commit
                        .into(),
                    &proofs,
                    NUM_CHILDREN_INTERNAL,
                );
                let group = format!("internal_verifier_height_{}", internal_node_height);
                proofs = info_span!("internal verifier", group = group).in_scope(|| {
                    internal_inputs
                        .into_iter()
                        .map(|input| {
                            let ret = info_span!(
                                "Internal verifier proof",
                                index = internal_node_idx,
                                height = internal_node_height
                            )
                            .in_scope(|| {
                                single_segment_execute_and_prove(&internal_prover, input.write())
                            });
                            internal_node_idx += 1;
                            ret
                        })
                        .collect()
                });
                internal_node_height += 1;
            }
            proofs.pop().unwrap()
        };
        #[allow(unused_variables)]
        let root_proof = info_span!("root verifier", group = "root_verifier").in_scope(move || {
            let root_prover = RootVerifierLocalProver::new(axvm_sdk_pk.root_verifier_pk.clone());
            let root_input = RootVmVerifierInput {
                proofs: vec![final_internal_proof],
                public_values: app_proofs.user_public_values.public_values,
            };
            let input = root_input.write();
            bench_root_verifier_prover(&root_prover, input.clone());
            SingleSegmentVmProver::prove(&root_prover, input)
        });
    });

    Ok(())
}

fn generate_fib_exe(app_fri_params: FriParameters) -> Arc<AxVmCommittedExe<SC>> {
    let elf = build_bench_program("fibonacci").unwrap();
    let mut exe: AxVmExe<_> = elf.into();
    println!("Program size: {}", exe.program.len());
    println!("Init memory size: {}", exe.init_memory.len());
    exe.program.max_num_public_values = NUM_PUBLIC_VALUES;

    let app_engine = BabyBearPoseidon2Engine::new(app_fri_params);
    Arc::new(AxVmCommittedExe::<SC>::commit(exe, app_engine.config.pcs()))
}

fn single_segment_execute_and_prove<SC: StarkGenericConfig, E: StarkFriEngine<SC>>(
    prover: &VmLocalProver<SC, E>,
    input: Vec<Vec<Val<SC>>>,
) -> Proof<SC>
where
    Val<SC>: PrimeField32,
{
    counter!("fri.log_blowup").absolute(prover.pk.fri_params.log_blowup as u64);
    let mut vm_config = prover.pk.vm_config.clone();
    vm_config.collect_metrics = true;
    let vm = SingleSegmentVmExecutor::new(vm_config);
    vm.execute(prover.committed_exe.exe.clone(), input.clone())
        .unwrap();
    SingleSegmentVmProver::prove(prover, input)
}

fn bench_root_verifier_prover(prover: &RootVerifierLocalProver, input: Vec<Vec<F>>) {
    counter!("fri.log_blowup").absolute(prover.root_verifier_pk.vm_pk.fri_params.log_blowup as u64);
    let mut vm_config = prover.root_verifier_pk.vm_pk.vm_config.clone();
    vm_config.collect_metrics = true;
    let vm = SingleSegmentVmExecutor::new(vm_config);
    vm.execute(
        prover.root_verifier_pk.root_committed_exe.exe.clone(),
        input,
    )
    .unwrap();
}
