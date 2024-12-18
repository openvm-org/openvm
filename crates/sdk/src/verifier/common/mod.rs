use std::array;

use openvm_circuit::{
    arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    },
    system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs},
};
use openvm_native_compiler::{ir::Config, prelude::*};
use openvm_native_recursion::{digest::DigestVariable, vars::StarkProofVariable};
use openvm_stark_sdk::openvm_stark_backend::p3_field::AbstractField;

use crate::verifier::internal::types::InternalVmVerifierPvs;

pub mod non_leaf;
pub mod types;

pub fn assert_or_assign_app_and_leaf_commit_pvs<C: Config>(
    builder: &mut Builder<C>,
    dst: &InternalVmVerifierPvs<Felt<C::F>>,
    proof_idx: RVar<C::N>,
    proof_pvs: &InternalVmVerifierPvs<Felt<C::F>>,
) {
    builder.if_eq(proof_idx, RVar::zero()).then_or_else(
        |builder| {
            builder.assign(
                &dst.vm_verifier_pvs.app_commit,
                proof_pvs.vm_verifier_pvs.app_commit,
            );
            builder.assign(
                &dst.extra_pvs.leaf_verifier_commit,
                proof_pvs.extra_pvs.leaf_verifier_commit,
            );
        },
        |builder| {
            builder.assert_eq::<[_; DIGEST_SIZE]>(
                dst.vm_verifier_pvs.app_commit,
                proof_pvs.vm_verifier_pvs.app_commit,
            );
            builder.assert_eq::<[_; DIGEST_SIZE]>(
                dst.extra_pvs.leaf_verifier_commit,
                proof_pvs.extra_pvs.leaf_verifier_commit,
            );
        },
    );
}

pub fn assert_or_assign_connector_pvs<C: Config>(
    builder: &mut Builder<C>,
    dst: &VmConnectorPvs<Felt<C::F>>,
    proof_idx: RVar<C::N>,
    proof_pvs: &VmConnectorPvs<Felt<C::F>>,
) {
    builder.if_eq(proof_idx, RVar::zero()).then_or_else(
        |builder| {
            builder.assign(&dst.initial_pc, proof_pvs.initial_pc);
        },
        |builder| {
            // assert prev.final_pc == curr.initial_pc
            builder.assert_felt_eq(dst.final_pc, proof_pvs.initial_pc);
            // assert prev.is_terminate == 0
            builder.assert_felt_eq(dst.is_terminate, C::F::ZERO);
        },
    );
    // Update final_pc
    builder.assign(&dst.final_pc, proof_pvs.final_pc);
    // Update is_terminate
    builder.assign(&dst.is_terminate, proof_pvs.is_terminate);
    // Update exit_code
    builder.assign(&dst.exit_code, proof_pvs.exit_code);
}

pub fn assert_or_assign_memory_pvs<C: Config>(
    builder: &mut Builder<C>,
    dst: &MemoryMerklePvs<Felt<C::F>, DIGEST_SIZE>,
    proof_idx: RVar<C::N>,
    proof_pvs: &MemoryMerklePvs<Felt<C::F>, DIGEST_SIZE>,
) {
    builder.if_eq(proof_idx, RVar::zero()).then_or_else(
        |builder| {
            builder.assign(&dst.initial_root, proof_pvs.initial_root);
        },
        |builder| {
            // assert prev.final_root == curr.initial_root
            builder.assert_eq::<[_; DIGEST_SIZE]>(dst.final_root, proof_pvs.initial_root);
        },
    );
    // Update final_root
    builder.assign(&dst.final_root, proof_pvs.final_root);
}

pub fn get_program_commit<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) -> [Felt<C::F>; DIGEST_SIZE] {
    let t_id = RVar::from(PROGRAM_CACHED_TRACE_INDEX);
    let commit = builder.get(&proof.commitments.main_trace, t_id);
    let commit = if let DigestVariable::Felt(commit) = commit {
        commit
    } else {
        unreachable!()
    };
    array::from_fn(|i| builder.get(&commit, i))
}

pub fn get_connector_pvs<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) -> VmConnectorPvs<Felt<C::F>> {
    get_connector_pvs_impl(builder, proof, CONNECTOR_AIR_ID)
}

fn get_connector_pvs_impl<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
    connector_air_id: usize,
) -> VmConnectorPvs<Felt<C::F>> {
    let a_id = RVar::from(connector_air_id);
    let a_input = builder.get(&proof.per_air, a_id);
    let proof_pvs = &a_input.public_values;
    VmConnectorPvs {
        initial_pc: builder.get(proof_pvs, 0),
        final_pc: builder.get(proof_pvs, 1),
        exit_code: builder.get(proof_pvs, 2),
        is_terminate: builder.get(proof_pvs, 3),
    }
}

pub fn get_memory_pvs<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) -> MemoryMerklePvs<Felt<C::F>, DIGEST_SIZE> {
    let a_id = RVar::from(MERKLE_AIR_ID);
    let a_input = builder.get(&proof.per_air, a_id);
    MemoryMerklePvs {
        initial_root: array::from_fn(|i| builder.get(&a_input.public_values, i)),
        final_root: array::from_fn(|i| builder.get(&a_input.public_values, i + DIGEST_SIZE)),
    }
}

/// Asserts that a single segment VM  exits successfully.
pub fn assert_single_segment_vm_exit_successfully<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) {
    assert_single_segment_vm_exit_successfully_with_connector_air_id(
        builder,
        proof,
        CONNECTOR_AIR_ID,
    )
}

pub fn assert_single_segment_vm_exit_successfully_with_connector_air_id<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
    connector_air_id: usize,
) {
    let connector_pvs = get_connector_pvs_impl(builder, proof, connector_air_id);
    // FIXME: does single segment VM program always have pc_start = 0?
    // Start PC should be 0
    builder.assert_felt_eq(connector_pvs.initial_pc, C::F::ZERO);
    // Terminate should be 1
    builder.assert_felt_eq(connector_pvs.is_terminate, C::F::ONE);
    // Exit code should be 0
    builder.assert_felt_eq(connector_pvs.exit_code, C::F::ZERO);
}

// TODO: This is a temporary solution. VK should be able to specify which AIRs are required. Once
// that is implemented, this function can be removed.
pub fn assert_required_air_for_agg_vm_present<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) {
    // FIXME: what if PUBLIC_VALUES_AIR_ID(3) >= proof.per_air.len()?
    let program_air = builder.get(&proof.per_air, PROGRAM_AIR_ID);
    builder.assert_eq::<Usize<_>>(program_air.air_id, RVar::from(PROGRAM_AIR_ID));
    let connector_air = builder.get(&proof.per_air, CONNECTOR_AIR_ID);
    builder.assert_eq::<Usize<_>>(connector_air.air_id, RVar::from(CONNECTOR_AIR_ID));
    let public_values_air = builder.get(&proof.per_air, PUBLIC_VALUES_AIR_ID);
    builder.assert_eq::<Usize<_>>(public_values_air.air_id, RVar::from(PUBLIC_VALUES_AIR_ID));
}

// TODO: This is a temporary solution. VK should be able to specify which AIRs are required. Once
// that is implemented, this function can be removed.
pub fn assert_required_air_for_app_vm_present<C: Config>(
    builder: &mut Builder<C>,
    proof: &StarkProofVariable<C>,
) {
    // FIXME: what if MERKLE_AIR_ID(4) >= proof.per_air.len()?
    let program_air = builder.get(&proof.per_air, PROGRAM_AIR_ID);
    builder.assert_eq::<Usize<_>>(program_air.air_id, RVar::from(PROGRAM_AIR_ID));
    let connector_air = builder.get(&proof.per_air, CONNECTOR_AIR_ID);
    builder.assert_eq::<Usize<_>>(connector_air.air_id, RVar::from(CONNECTOR_AIR_ID));
    let public_values_air = builder.get(&proof.per_air, MERKLE_AIR_ID);
    builder.assert_eq::<Usize<_>>(public_values_air.air_id, RVar::from(MERKLE_AIR_ID));
}
