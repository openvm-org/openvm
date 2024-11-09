use std::{array, borrow::Borrow};

use ax_stark_sdk::ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::arch::PUBLIC_VALUES_AIR_ID;
use axvm_native_compiler::ir::{Array, Builder, Config, Felt, RVar, Usize, DIGEST_SIZE};
use axvm_recursion::{
    challenger::duplex::DuplexChallengerVariable, fri::TwoAdicFriPcsVariable, stark::StarkVerifier,
    types::MultiStarkVerificationAdvice, vars::StarkProofVariable,
};

use crate::verifier::{
    common::{
        assert_or_assign_connector_pvs, assert_or_assign_memory_pvs,
        assert_single_segment_vm_exit_successfully, get_program_commit, types::VmVerifierPvs,
    },
    internal::types::InternalVmVerifierPvs,
    utils::{assign_array_to_slice, eq_felt_slice},
};

pub struct NonLeafVerifierVariables<C: Config> {
    pub internal_program_commit: [Felt<C::F>; DIGEST_SIZE],
    pub leaf_pcs: TwoAdicFriPcsVariable<C>,
    pub leaf_advice: MultiStarkVerificationAdvice<C>,
    pub internal_pcs: TwoAdicFriPcsVariable<C>,
    pub internal_advice: MultiStarkVerificationAdvice<C>,
}

impl<C: Config> NonLeafVerifierVariables<C> {
    /// Verify proofs of internal verifier or leaf verifier.
    /// Returns aggregated VmVerifierPvs and leaf verifier commitment of these proofs.
    #[allow(clippy::type_complexity)]
    pub fn verify_internal_or_leaf_verifier_proofs(
        &self,
        builder: &mut Builder<C>,
        proofs: &Array<C, StarkProofVariable<C>>,
    ) -> (VmVerifierPvs<Felt<C::F>>, [Felt<C::F>; DIGEST_SIZE])
    where
        C::F: PrimeField32,
    {
        // At least 1 proof should be provided.
        builder.assert_ne::<Usize<_>>(proofs.len(), RVar::zero());
        let pvs = VmVerifierPvs::<Felt<C::F>>::uninit(builder);
        let leaf_verifier_commit = array::from_fn(|_| builder.uninit());

        builder.range(0, proofs.len()).for_each(|i, builder| {
            let proof = builder.get(proofs, i);
            let proof_vm_pvs = self.verify_internal_or_leaf_verifier_proof(builder, &proof);

            assert_single_segment_vm_exit_successfully(builder, &proof);
            builder.if_eq(i, RVar::zero()).then_or_else(
                |builder| {
                    builder.assign(&pvs.app_commit, proof_vm_pvs.vm_verifier_pvs.app_commit);
                    builder.assign(
                        &leaf_verifier_commit,
                        proof_vm_pvs.extra_pvs.leaf_verifier_commit,
                    );
                },
                |builder| {
                    builder.assert_eq::<[_; DIGEST_SIZE]>(
                        pvs.app_commit,
                        proof_vm_pvs.vm_verifier_pvs.app_commit,
                    );
                    builder.assert_eq::<[_; DIGEST_SIZE]>(
                        leaf_verifier_commit,
                        proof_vm_pvs.extra_pvs.leaf_verifier_commit,
                    );
                },
            );
            assert_or_assign_connector_pvs(
                builder,
                &pvs.connector,
                i,
                &proof_vm_pvs.vm_verifier_pvs.connector,
            );
            assert_or_assign_memory_pvs(
                builder,
                &pvs.memory,
                i,
                &proof_vm_pvs.vm_verifier_pvs.memory,
            );
            // This is only needed when `is_terminate` but branching here won't save much, so we
            // always assign it.
            builder.assign(
                &pvs.public_values_commit,
                proof_vm_pvs.vm_verifier_pvs.public_values_commit,
            );
        });
        (pvs, leaf_verifier_commit)
    }
    fn verify_internal_or_leaf_verifier_proof(
        &self,
        builder: &mut Builder<C>,
        proof: &StarkProofVariable<C>,
    ) -> InternalVmVerifierPvs<Felt<C::F>>
    where
        C::F: PrimeField32,
    {
        let flatten_proof_vm_pvs = InternalVmVerifierPvs::<Felt<C::F>>::uninit(builder).flatten();
        let proof_vm_pvs_arr = builder
            .get(&proof.per_air, PUBLIC_VALUES_AIR_ID)
            .public_values;

        let program_commit = get_program_commit(builder, proof);
        let is_self_program =
            eq_felt_slice(builder, &self.internal_program_commit, &program_commit);

        builder.if_eq(is_self_program, RVar::one()).then_or_else(
            |builder| {
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder,
                    &self.internal_pcs,
                    &self.internal_advice,
                    proof,
                );
                assign_array_to_slice(builder, &flatten_proof_vm_pvs, &proof_vm_pvs_arr, 0);
                let proof_vm_pvs: &InternalVmVerifierPvs<_> =
                    flatten_proof_vm_pvs.as_slice().borrow();
                // Handle recursive verification
                // For proofs, its program commitment should be committed.
                builder.assert_eq::<[_; DIGEST_SIZE]>(
                    proof_vm_pvs.extra_pvs.internal_program_commit,
                    program_commit,
                );
            },
            |builder| {
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder,
                    &self.leaf_pcs,
                    &self.leaf_advice,
                    proof,
                );
                // Leaf verifier doesn't have extra public values.
                assign_array_to_slice(
                    builder,
                    &flatten_proof_vm_pvs[..VmVerifierPvs::<u8>::width()],
                    &proof_vm_pvs_arr,
                    0,
                );
                let proof_vm_pvs: &InternalVmVerifierPvs<_> =
                    flatten_proof_vm_pvs.as_slice().borrow();
                builder.assign(&proof_vm_pvs.extra_pvs.leaf_verifier_commit, program_commit);
            },
        );
        *flatten_proof_vm_pvs.as_slice().borrow()
    }
}
