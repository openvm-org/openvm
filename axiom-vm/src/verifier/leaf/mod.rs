use std::{array, borrow::BorrowMut};

use afs_derive::AlignedBorrow;
use ax_sdk::{
    ax_stark_backend::{
        keygen::types::MultiStarkVerifyingKey, p3_field::AbstractField, prover::types::Proof,
    },
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
};
use axvm_circuit::{
    arch::instructions::program::Program,
    system::{
        connector::VmConnectorPvs,
        memory::merkle::MemoryMerklePvs,
        vm::{
            chip_set::{CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_CACHED_TRACE_INDEX},
            config::VmConfig,
        },
    },
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{
    challenger::duplex::DuplexChallengerVariable,
    digest::DigestVariable,
    fri::TwoAdicFriPcsVariable,
    hints::{Hintable, InnerVal},
    stark::StarkVerifier,
    types::{new_from_inner_multi_vk, InnerConfig},
    utils::const_fri_config,
    vars::StarkProofVariable,
};

use crate::config::AxiomVmConfig;

type C = InnerConfig;
type F = InnerVal;

#[derive(Debug, AlignedBorrow)]
#[repr(C)]
pub struct LeafVmVerifierPvs<T, const CHUNK: usize> {
    // TODO: is to right to assume a trace commitment [T; CHUNK]?
    pub app_pc_start: T,
    pub app_commit: [T; CHUNK],
    pub connector: VmConnectorPvs<T>,
    pub memory: MemoryMerklePvs<T, CHUNK>,
    pub public_values_commit: [T; CHUNK],
}

impl<const CHUNK: usize> LeafVmVerifierPvs<Felt<F>, { CHUNK }> {
    fn uninit(builder: &mut Builder<C>) -> Self {
        Self {
            app_pc_start: builder.uninit(),
            app_commit: array::from_fn(|_| builder.uninit()),
            connector: VmConnectorPvs {
                initial_pc: builder.uninit(),
                final_pc: builder.uninit(),
                exit_code: builder.uninit(),
                is_terminate: builder.uninit(),
            },
            memory: MemoryMerklePvs {
                initial_root: array::from_fn(|_| builder.uninit()),
                final_root: array::from_fn(|_| builder.uninit()),
            },
            public_values_commit: array::from_fn(|_| builder.uninit()),
        }
    }
}

impl<const CHUNK: usize> LeafVmVerifierPvs<Felt<F>, { CHUNK }> {
    pub fn flatten(self) -> Vec<Felt<F>> {
        let mut v = vec![Felt(0, Default::default()); LeafVmVerifierPvs::<u8, CHUNK>::width()];
        *v.as_mut_slice().borrow_mut() = self;
        v
    }
}

/// Config to generate
pub struct LeafVmVerifierConfig {
    #[allow(unused)]
    pub max_num_user_public_values: usize,
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}

impl LeafVmVerifierConfig {
    pub fn build_program(
        self,
        app_vm_vk: MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        let m_advice = new_from_inner_multi_vk(&app_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.fri_params),
            };
            let proofs: Array<C, StarkProofVariable<_>> =
                <Vec<Proof<BabyBearPoseidon2Config>> as Hintable<C>>::read(&mut builder);
            // At least 1 proof should be provided.
            builder.assert_ne::<Usize<_>>(proofs.len(), RVar::zero());

            let pvs = LeafVmVerifierPvs::<Felt<F>, { DIGEST_SIZE }>::uninit(&mut builder);
            builder.range(0, proofs.len()).for_each(|i, builder| {
                let proof = builder.get(&proofs, i);
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder, &pcs, &m_advice, &proof,
                );
                {
                    // TODO: Add app_pc_start
                    builder.assign(&pvs.app_pc_start, F::zero());
                    let t_id = RVar::from(PROGRAM_CACHED_TRACE_INDEX);
                    let commit = builder.get(&proof.commitments.main_trace, t_id);
                    let commit = if let DigestVariable::Felt(commit) = commit {
                        commit
                    } else {
                        unreachable!()
                    };
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| assign_slice(builder, &pvs.app_commit, &commit, 0),
                        |builder| assert_slice(builder, &pvs.app_commit, &commit, 0),
                    );
                }
                {
                    let a_id = RVar::from(CONNECTOR_AIR_ID);
                    let a_input = builder.get(&proof.per_air, a_id);
                    let connector_pvs = &pvs.connector;
                    let input_pvs = &a_input.public_values;
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| assign_connector(builder, connector_pvs, input_pvs),
                        |builder| {
                            // assert prev.final_pc == curr.initial_pc
                            let initial_pc = builder.get(input_pvs, 0);
                            builder.assert_felt_eq(connector_pvs.final_pc, initial_pc);
                            // Update final_pc
                            let final_pc = builder.get(input_pvs, 1);
                            builder.assign(&connector_pvs.final_pc, final_pc);
                            // assert prev.is_terminate == 0
                            builder.assert_felt_eq(connector_pvs.is_terminate, F::zero());
                            // Update is_terminate
                            let is_terminate = builder.get(input_pvs, 3);
                            builder.assign(&connector_pvs.is_terminate, is_terminate);
                            // Update exit_code
                            let exit_code = builder.get(input_pvs, 2);
                            builder.assign(&connector_pvs.exit_code, exit_code);
                        },
                    );
                }
                {
                    let a_id = RVar::from(MERKLE_AIR_ID);
                    let a_input = builder.get(&proof.per_air, a_id);
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| {
                            assign_slice(
                                builder,
                                &pvs.memory.initial_root,
                                &a_input.public_values,
                                0,
                            );
                            assign_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                DIGEST_SIZE,
                            );
                        },
                        |builder| {
                            // assert prev.final_root == curr.initial_root
                            assert_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                0,
                            );
                            // Update final_root
                            assign_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                DIGEST_SIZE,
                            );
                        },
                    );
                }
                // TODO: decommit user public value address space.
                for j in 0..DIGEST_SIZE {
                    builder.assign(&pvs.public_values_commit[j], F::zero());
                }
            });
            for pv in pvs.flatten() {
                builder.commit_public_value(pv);
            }

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options)
    }
}

impl AxiomVmConfig {
    pub fn leaf_verifier_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            LeafVmVerifierPvs::<u8, DIGEST_SIZE>::width(),
            self.poseidon2_max_constraint_degree,
        )
    }
}

fn assign_slice<const CHUNK: usize>(
    builder: &mut Builder<C>,
    dst_slice: &[Felt<F>; CHUNK],
    src: &Array<C, Felt<F>>,
    src_offset: usize,
) {
    for (i, dst) in dst_slice.iter().enumerate() {
        let pv = builder.get(src, i + src_offset);
        builder.assign(dst, pv);
    }
}

fn assert_slice<const CHUNK: usize>(
    builder: &mut Builder<C>,
    dst_slice: &[Felt<F>; CHUNK],
    src: &Array<C, Felt<F>>,
    src_offset: usize,
) {
    for (i, &dst) in dst_slice.iter().enumerate() {
        let pv = builder.get(src, i + src_offset);
        builder.assert_felt_eq(dst, pv);
    }
}

fn assign_connector(
    builder: &mut Builder<C>,
    dst: &VmConnectorPvs<Felt<F>>,
    src: &Array<C, Felt<F>>,
) {
    let VmConnectorPvs {
        initial_pc,
        final_pc,
        exit_code,
        is_terminate,
    } = dst;
    let v = builder.get(src, RVar::from(0));
    builder.assign(initial_pc, v);
    let v = builder.get(src, RVar::from(1));
    builder.assign(final_pc, v);
    let v = builder.get(src, RVar::from(2));
    builder.assign(exit_code, v);
    let v = builder.get(src, RVar::from(3));
    builder.assign(is_terminate, v);
}
