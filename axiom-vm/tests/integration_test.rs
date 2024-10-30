use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, p3_field::AbstractField},
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axiom_vm::config::{AxiomVmConfig, AxiomVmProvingKey};
use axvm_circuit::{
    arch::{
        ExecutorName, MemoryConfig, PersistenceType, SingleSegmentVmExecutor, VmConfig, VmExecutor,
    },
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{hints::Hintable, types::InnerConfig};
use p3_baby_bear::BabyBear;

type C = InnerConfig;
type F = BabyBear;
#[test]
fn test_1() {
    let axiom_vm_config = AxiomVmConfig {
        poseidon2_max_constraint_degree: 7,
        max_num_user_public_values: 100,
        fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
        app_vm_config: VmConfig {
            max_segment_len: 200,
            memory_config: MemoryConfig {
                persistence_type: PersistenceType::Persistent,
                ..Default::default()
            },
            ..Default::default()
        }
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::FieldArithmetic),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            compile_prints: true,
            ..Default::default()
        },
    };
    let axiom_vm_pk = AxiomVmProvingKey::keygen(axiom_vm_config);
    let engine = BabyBearPoseidon2Engine::new(axiom_vm_pk.fri_params);

    let program = {
        let n = 100;
        let mut builder = Builder::<C>::default();
        let a: Felt<F> = builder.eval(F::zero());
        let b: Felt<F> = builder.eval(F::one());
        let c: Felt<F> = builder.uninit();
        builder.range(0, n).for_each(|_, builder| {
            builder.assign(&c, a + b);
            builder.assign(&a, b);
            builder.assign(&b, c);
        });
        builder.halt();
        builder.compile_isa()
    };
    let committed_exe = Arc::new(AxVmCommittedExe::commit(
        program.into(),
        engine.config.pcs(),
    ));

    let app_vm = VmExecutor::new(axiom_vm_pk.app_vm_config.clone());
    let app_vm_result = app_vm
        .execute_and_generate_with_cached_program(committed_exe, vec![])
        .unwrap();
    assert!(app_vm_result.per_segment.len() > 1);
    let app_vm_seg_proofs: Vec<_> = app_vm_result
        .per_segment
        .into_iter()
        .map(|proof_input| engine.prove(&axiom_vm_pk.app_vm_pk, proof_input))
        .collect();

    let leaf_vm = SingleSegmentVmExecutor::new(axiom_vm_pk.leaf_vm_config);
    leaf_vm
        .execute(
            axiom_vm_pk.committed_leaf_program.exe.clone(),
            app_vm_seg_proofs.write(),
        )
        .unwrap();
}
