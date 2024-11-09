use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::config::StarkGenericConfig,
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axiom_vm::{
    config::{AxiomVmConfig, AxiomVmProvingKey},
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
};
use axvm_circuit::{
    arch::{instructions::program::Program, ExecutorName, VmConfig},
    prover::{local::VmLocalProver, ContinuationVmProver, SingleSegmentVmProver},
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_compiler::{
    conversion::CompilerOptions,
    ir::{Builder, Felt},
};
use axvm_recursion::{hints::Hintable, types::InnerConfig};
use eyre::Result;
use p3_field::AbstractField;
use tracing::info_span;

type OuterSC = BabyBearPoseidon2OuterConfig;
type SC = BabyBearPoseidon2Config;
type C = InnerConfig;
type F = BabyBear;
const NUM_PUBLIC_VALUES: usize = 16;
const NUM_CHILDREN_LEAF: usize = 2;
const NUM_CHILDREN_INTERNAL: usize = 2;

#[tokio::main]

async fn main() -> Result<()> {
    let num_segments = 8;
    let segment_len = 100000;
    let axiom_vm_pk = {
        let axiom_vm_config = AxiomVmConfig {
            max_num_user_public_values: 16,
            app_fri_params: standard_fri_params_with_100_bits_conjectured_security(1),
            leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(2),
            internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
            root_fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
            app_vm_config: VmConfig {
                poseidon2_max_constraint_degree: 3,
                max_segment_len: segment_len,
                continuation_enabled: true,
                num_public_values: NUM_PUBLIC_VALUES,
                ..Default::default()
            }
            .add_executor(ExecutorName::BranchEqual)
            .add_executor(ExecutorName::Jal)
            .add_executor(ExecutorName::LoadStore)
            .add_executor(ExecutorName::FieldArithmetic),
            compiler_options: CompilerOptions {
                enable_cycle_tracker: true,
                ..Default::default()
            },
        };
        AxiomVmProvingKey::keygen(axiom_vm_config)
    };

    let app_committed_exe = generate_fib_exe(axiom_vm_pk.app_fri_params, num_segments, segment_len);
    run_with_metric_collection("OUTPUT_PATH", || {
        let app_proofs = info_span!("App VM", group = "app_vm").in_scope(|| {
            let app_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                axiom_vm_pk.app_fri_params,
                axiom_vm_pk.app_vm_config.clone(),
                axiom_vm_pk.app_vm_pk.clone(),
                app_committed_exe.clone(),
            );
            ContinuationVmProver::prove(&app_prover, vec![])
        });

        let leaf_proofs = info_span!("leaf verifier", group = "leaf_verifier").in_scope(|| {
            let leaf_inputs =
                LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proofs, NUM_CHILDREN_LEAF);
            let leaf_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                axiom_vm_pk.leaf_fri_params,
                axiom_vm_pk.leaf_vm_config.clone(),
                axiom_vm_pk.leaf_vm_pk.clone(),
                axiom_vm_pk.leaf_committed_exe.clone(),
            );
            leaf_inputs
                .into_iter()
                .enumerate()
                .map(|(leaf_idx, input)| {
                    info_span!("leaf verifier proof", index = leaf_idx).in_scope(|| {
                        SingleSegmentVmProver::prove(&leaf_prover, input.write_to_stream())
                    })
                })
                .collect::<Vec<_>>()
        });
        let final_internal_proof = info_span!("internal verifier", group = "internal_verifier")
            .in_scope(|| {
                let internal_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
                    axiom_vm_pk.internal_fri_params,
                    axiom_vm_pk.internal_vm_config.clone(),
                    axiom_vm_pk.internal_vm_pk.clone(),
                    axiom_vm_pk.internal_committed_exe.clone(),
                );
                let mut internal_node_idx = 0;
                let mut internal_node_height = 0;
                let mut proofs = leaf_proofs;
                while proofs.len() > 1 {
                    let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                        axiom_vm_pk
                            .internal_committed_exe
                            .committed_program
                            .prover_data
                            .commit
                            .into(),
                        &proofs,
                        NUM_CHILDREN_INTERNAL,
                    );
                    proofs = internal_inputs
                        .into_iter()
                        .map(|input| {
                            let ret = info_span!(
                                "Internal verifier proof",
                                index = internal_node_idx,
                                height = internal_node_height
                            )
                            .in_scope(|| {
                                SingleSegmentVmProver::prove(&internal_prover, input.write())
                            });
                            internal_node_idx += 1;
                            ret
                        })
                        .collect();
                    internal_node_height += 1;
                }
                proofs.pop().unwrap()
            });
        #[allow(unused_variables)]
        let root_proof = info_span!("root verifier", group = "root_verifier").in_scope(move || {
            let root_prover = VmLocalProver::<OuterSC, BabyBearPoseidon2OuterEngine>::new(
                axiom_vm_pk.root_fri_params,
                axiom_vm_pk.root_vm_config.clone(),
                axiom_vm_pk.root_vm_pk.clone(),
                axiom_vm_pk.root_committed_exe.clone(),
            );
            let root_input = RootVmVerifierInput {
                proofs: vec![final_internal_proof],
                public_values: app_proofs.user_public_values.public_values,
            };
            SingleSegmentVmProver::prove(&root_prover, root_input.write())
        });
    });

    Ok(())
}

fn generate_fib_exe(
    app_fri_params: FriParameters,
    num_segments: usize,
    segment_len: usize,
) -> Arc<AxVmCommittedExe<SC>> {
    let program = generate_fib_program(num_segments, segment_len);
    let app_engine = BabyBearPoseidon2Engine::new(app_fri_params);
    Arc::new(AxVmCommittedExe::<SC>::commit(
        program.into(),
        app_engine.config.pcs(),
    ))
}
fn generate_fib_program(num_segments: usize, segment_len: usize) -> Program<F> {
    let total_cycles = num_segments * segment_len;
    let mut program = {
        // 3 instructions: initialize a,b/ halt.
        // 4 instructions per iteration.
        let n = (total_cycles - 3) / 3;
        let mut builder = Builder::<C>::default();
        let a: Felt<F> = builder.eval(F::ZERO);
        let b: Felt<F> = builder.eval(F::ONE);
        let c: Felt<F> = builder.uninit();
        builder.range(0, n).for_each(|_, builder| {
            builder.assign(&c, a + b);
            builder.assign(&a, b);
            builder.assign(&b, c);
        });
        builder.halt();
        builder.compile_isa()
    };
    program.max_num_public_values = 0;
    program
}
