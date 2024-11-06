use std::array;

use ax_stark_sdk::ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::{
    arch::{CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_CACHED_TRACE_INDEX},
    system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs},
};
use axvm_native_compiler::{ir::Config, prelude::*};
use axvm_recursion::{digest::DigestVariable, vars::StarkProofVariable};

pub mod types;

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
    let a_id = RVar::from(CONNECTOR_AIR_ID);
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
    let connector_pvs = get_connector_pvs(builder, proof);
    // FIXME: does single segment VM program always have pc_start = 0?
    // Start PC should be 0
    builder.assert_felt_eq(connector_pvs.initial_pc, C::F::ZERO);
    // Terminate should be 1
    builder.assert_felt_eq(connector_pvs.is_terminate, C::F::ONE);
    // Exit code should be 0
    builder.assert_felt_eq(connector_pvs.exit_code, C::F::ZERO);
}
