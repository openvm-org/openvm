use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::system::{
    connector::{DEFAULT_SUSPEND_EXIT_CODE, VmConnectorPvs},
    memory::merkle::MemoryMerklePvs,
};
use openvm_circuit_primitives::utils::{and, assert_array_eq, not, select};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use recursion_circuit::{
    bus::{CachedCommitBus, CachedCommitBusMessage, PublicValuesBus, PublicValuesBusMessage},
    utils::assert_zeros,
};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::public_values::{NonRootVerifierPvs, app::*};

pub const VERIFIER_PVS_AIR_ID: usize = 0;
pub const CONSTRAINT_EVAL_AIR_ID: usize = 1;
pub const CONSTRAINT_EVAL_CACHED_INDEX: usize = 0;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierPvsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_last: F,
    pub has_verifier_pvs: F,
    pub child_pvs: NonRootVerifierPvs<F>,
}

pub enum VerifierChildLevel {
    App,
    Leaf,
    InternalForLeaf,
    InternalRecursive,
}

pub fn generate_proving_ctx(
    proofs: &[Proof],
    user_pv_commit: Option<[F; DIGEST_SIZE]>,
    child_vk_commit: [F; DIGEST_SIZE],
) -> AirProvingContextV2<CpuBackendV2> {
    let num_proofs = proofs.len();
    let height = num_proofs.next_power_of_two();
    let width = VerifierPvsCols::<u8>::width();

    debug_assert!(num_proofs > 0);

    let mut trace = vec![F::ZERO; height * width];
    let mut child_level = VerifierChildLevel::App;

    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut VerifierPvsCols<F> = chunk.borrow_mut();

        cols.proof_idx = F::from_canonical_usize(proof_idx);
        cols.is_valid = F::ONE;
        cols.is_last = F::from_bool(proof_idx + 1 == num_proofs);

        if let Some(user_pv_commit) = user_pv_commit {
            cols.has_verifier_pvs = F::ZERO;
            cols.child_pvs.user_pv_commit = user_pv_commit;
            cols.child_pvs.program_commit = proof.trace_vdata[PROGRAM_AIR_ID]
                .as_ref()
                .unwrap()
                .cached_commitments[PROGRAM_CACHED_TRACE_INDEX];

            let &VmConnectorPvs {
                initial_pc,
                final_pc,
                exit_code,
                is_terminate,
            } = proof.public_values[CONNECTOR_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_pc = initial_pc;
            cols.child_pvs.final_pc = final_pc;
            cols.child_pvs.exit_code = exit_code;
            cols.child_pvs.is_terminate = is_terminate;

            let &MemoryMerklePvs::<_, DIGEST_SIZE> {
                initial_root,
                final_root,
            } = proof.public_values[MERKLE_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_root = initial_root;
            cols.child_pvs.final_root = final_root;
        } else {
            let child_pvs: &NonRootVerifierPvs<F> =
                proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
            cols.has_verifier_pvs = F::ONE;
            cols.child_pvs = *child_pvs;

            child_level = match child_pvs.internal_flag {
                F::ZERO => VerifierChildLevel::Leaf,
                F::ONE => VerifierChildLevel::InternalForLeaf,
                F::TWO => VerifierChildLevel::InternalRecursive,
                _ => unreachable!(),
            }
        }
    }

    let last_row: &VerifierPvsCols<F> =
        trace[(proofs.len() - 1) * width..proofs.len() * width].borrow();
    let mut pvs = last_row.child_pvs;

    let first_row: &VerifierPvsCols<F> = trace[..width].borrow();
    pvs.initial_pc = first_row.child_pvs.initial_pc;
    pvs.initial_root = first_row.child_pvs.initial_root;

    match child_level {
        VerifierChildLevel::App => {
            pvs.leaf_commit = child_vk_commit;
        }
        VerifierChildLevel::Leaf => {
            pvs.internal_for_leaf_commit = child_vk_commit;
            pvs.internal_flag = F::ONE;
        }
        VerifierChildLevel::InternalForLeaf => {
            pvs.internal_recursive_commit = child_vk_commit;
            pvs.internal_flag = F::TWO;
        }
        VerifierChildLevel::InternalRecursive => {
            pvs.internal_flag = F::TWO;
        }
    }

    AirProvingContextV2 {
        cached_mains: vec![],
        common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
        public_values: pvs.to_vec(),
    }
}

pub struct VerifierPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
}

impl<F> BaseAir<F> for VerifierPvsAir {
    fn width(&self) -> usize {
        VerifierPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for VerifierPvsAir {
    fn num_public_values(&self) -> usize {
        NonRootVerifierPvs::<u8>::width()
    }
}
impl<F> PartitionedBaseAir<F> for VerifierPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for VerifierPvsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &VerifierPvsCols<AB::Var> = (*local).borrow();
        let next: &VerifierPvsCols<AB::Var> = (*next).borrow();

        /*
         * We first constrain segment adjacency, i.e. that rows in the trace are such that the
         * first row is the (chronologically) first segment, and adjacent rows correspond to
         * adjacent segments.
         */
        // constrain increasing proof_idx
        builder.assert_bool(local.is_valid);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when(and::<AB::Expr>(
                and(local.is_valid, next.is_valid),
                not(local.is_last),
            ))
            .assert_eq(local.proof_idx + AB::F::ONE, next.proof_idx);

        // constrain is_last, note proof_idx on the first row is 0
        builder.assert_bool(local.is_last);
        builder.when(local.is_last).assert_one(local.is_valid);
        builder
            .when(local.is_last)
            .assert_zero(next.is_valid * next.proof_idx);

        // constrain that is_terminate is the last valid proof
        builder.assert_bool(local.child_pvs.is_terminate);
        builder
            .when(local.child_pvs.is_terminate)
            .assert_one(local.is_last);
        builder
            .when(local.child_pvs.is_terminate)
            .assert_zero(local.child_pvs.exit_code);

        // constrain that non-terminal segments exited successfully
        builder.when(not(local.child_pvs.is_terminate)).assert_eq(
            local.child_pvs.exit_code,
            AB::F::from_canonical_u32(DEFAULT_SUSPEND_EXIT_CODE),
        );

        // when local and next are valid, constrain increasing proof_idx and adjacency
        let mut when_both = builder.when(and(local.is_valid, not(local.is_last)));
        when_both.assert_eq(local.child_pvs.final_pc, next.child_pvs.initial_pc);
        assert_array_eq(
            &mut when_both,
            local.child_pvs.final_root,
            next.child_pvs.initial_root,
        );

        /*
         * Next we constrain the consistency of verifier-specific public values. Note we can
         * determine what layer a verifier is at using the has_verifier_pvs and internal_flag
         * columns. If has_verifier_pvs is 0, then we have a leaf verifier that is verifying
         * an app segment proof. If has_verifier_pvs is 1 and internal_flag is 0, then we have
         * an internal verifier that has leaf children, and if internal_flag is 1 then we have
         * internal_for_leaf children. Finally, if internal_flag is 2 then we are verifying
         * internal_recursion children.
         */
        // constrain verifier pvs columns are the same across all rows
        when_both.assert_eq(local.child_pvs.internal_flag, next.child_pvs.internal_flag);
        assert_array_eq(
            &mut when_both,
            local.child_pvs.leaf_commit,
            next.child_pvs.leaf_commit,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.internal_for_leaf_commit,
            next.child_pvs.internal_for_leaf_commit,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.internal_recursive_commit,
            next.child_pvs.internal_recursive_commit,
        );

        // constraints for basic validity
        builder.assert_bool(local.has_verifier_pvs);
        builder.assert_tern(local.child_pvs.internal_flag);
        builder
            .when(local.has_verifier_pvs)
            .assert_one(local.is_valid);

        // constrain that child commits are 0 when they shouldn't be defined
        builder
            .when(not(local.has_verifier_pvs))
            .assert_zero(local.child_pvs.internal_flag);

        assert_zeros(
            &mut builder.when(not(local.has_verifier_pvs)),
            local.child_pvs.leaf_commit,
        );
        assert_zeros(
            &mut builder.when(
                (local.child_pvs.internal_flag - AB::F::ONE)
                    * (local.child_pvs.internal_flag - AB::F::TWO),
            ),
            local.child_pvs.internal_for_leaf_commit,
        );
        assert_zeros(
            &mut builder.when(local.child_pvs.internal_flag - AB::F::TWO),
            local.child_pvs.internal_recursive_commit,
        );

        /*
         * We need to receive public values from ProofShapeModule to ensure the values being read
         * here are correct. The leaf verifier needs to read public values from CONNECTOR_AIR_ID
         * and MERKLE_AIR_ID, while the internal verifier reads from VERIFIER_PVS_AIR_ID.
         */
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;

        let verifier_pvs_id = AB::F::from_canonical_usize(VERIFIER_PVS_AIR_ID);
        let verifier_pvs_id_cond = is_internal.clone() * verifier_pvs_id;

        let connector_id_cond = AB::Expr::from_canonical_usize(CONNECTOR_AIR_ID) * is_leaf.clone();
        let connector_pvs_offset =
            is_internal.clone() * AB::F::from_canonical_usize(2 * DIGEST_SIZE);

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: connector_id_cond.clone() + verifier_pvs_id_cond.clone(),
                pv_idx: connector_pvs_offset.clone(),
                value: local.child_pvs.initial_pc.into(),
            },
            local.is_valid,
        );

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: connector_id_cond.clone() + verifier_pvs_id_cond.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::ONE,
                value: local.child_pvs.final_pc.into(),
            },
            local.is_valid,
        );

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: connector_id_cond.clone() + verifier_pvs_id_cond.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::TWO,
                value: local.child_pvs.exit_code.into(),
            },
            local.is_valid,
        );

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: connector_id_cond.clone() + verifier_pvs_id_cond.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::from_canonical_u8(3),
                value: local.child_pvs.is_terminate.into(),
            },
            local.is_valid,
        );

        let merkle_id_cond = AB::Expr::from_canonical_usize(MERKLE_AIR_ID) * is_leaf.clone();
        let merkle_pvs_offset =
            connector_pvs_offset + is_internal.clone() * AB::Expr::from_canonical_u8(4);
        let verifier_pvs_offset =
            merkle_pvs_offset.clone() + AB::F::from_canonical_usize(2 * DIGEST_SIZE);

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.into(),
                pv_idx: verifier_pvs_offset.clone(),
                value: local.child_pvs.internal_flag.into(),
            },
            local.is_valid * is_internal.clone(),
        );

        for didx in 0..DIGEST_SIZE {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.into(),
                    pv_idx: AB::Expr::from_canonical_usize(didx),
                    value: local.child_pvs.user_pv_commit[didx].into(),
                },
                local.is_valid * is_internal.clone(),
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.into(),
                    pv_idx: AB::Expr::from_canonical_usize(didx + DIGEST_SIZE),
                    value: local.child_pvs.program_commit[didx].into(),
                },
                local.is_valid * is_internal.clone(),
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: merkle_id_cond.clone() + verifier_pvs_id_cond.clone(),
                    pv_idx: merkle_pvs_offset.clone() + AB::F::from_canonical_usize(didx),
                    value: local.child_pvs.initial_root[didx].into(),
                },
                local.is_valid,
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: merkle_id_cond.clone() + verifier_pvs_id_cond.clone(),
                    pv_idx: merkle_pvs_offset.clone()
                        + AB::F::from_canonical_usize(didx + DIGEST_SIZE),
                    value: local.child_pvs.final_root[didx].into(),
                },
                local.is_valid,
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.into(),
                    pv_idx: verifier_pvs_offset.clone() + AB::F::from_canonical_usize(didx + 1),
                    value: local.child_pvs.leaf_commit[didx].into(),
                },
                local.is_valid * is_internal.clone(),
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.into(),
                    pv_idx: verifier_pvs_offset.clone()
                        + AB::F::from_canonical_usize(didx + DIGEST_SIZE + 1),
                    value: local.child_pvs.internal_for_leaf_commit[didx].into(),
                },
                local.is_valid * is_internal.clone(),
            );

            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.into(),
                    pv_idx: verifier_pvs_offset.clone()
                        + AB::F::from_canonical_usize(didx + 2 * DIGEST_SIZE + 1),
                    value: local.child_pvs.internal_recursive_commit[didx].into(),
                },
                local.is_valid * is_internal.clone(),
            );
        }

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. Note that we only impose constraints for layers below the current
         * one - it is impossible for the current layer to know its own commit, and future layers
         * will catch if we pre-emptively define a current or future verifier commit.
         */
        let &NonRootVerifierPvs::<_> {
            initial_pc,
            user_pv_commit,
            program_commit: app_commit,
            is_terminate,
            final_pc,
            exit_code,
            initial_root,
            final_root,
            internal_flag,
            leaf_commit,
            internal_for_leaf_commit,
            internal_recursive_commit,
        } = builder.public_values().borrow();

        // constrain first proof pvs
        builder.when_first_row().assert_one(local.is_valid);
        builder
            .when_first_row()
            .assert_eq(local.child_pvs.initial_pc, initial_pc);
        assert_array_eq(
            &mut builder.when_first_row(),
            local.child_pvs.initial_root,
            initial_root,
        );

        // constrain last proof pvs
        builder
            .when(local.is_last)
            .assert_eq(local.child_pvs.is_terminate, is_terminate);
        builder
            .when(local.is_last)
            .assert_eq(local.child_pvs.final_pc, final_pc);
        assert_array_eq(
            &mut builder.when(local.is_last),
            local.child_pvs.final_root,
            final_root,
        );

        // constrain other app pvs
        builder
            .when(local.is_last)
            .assert_eq(local.child_pvs.exit_code, exit_code);
        assert_array_eq(builder, local.child_pvs.user_pv_commit, user_pv_commit);
        assert_array_eq(builder, local.child_pvs.program_commit, app_commit);

        // constrain internal_flag is 0 at the leaf level
        builder
            .when(not(local.has_verifier_pvs))
            .assert_zero(internal_flag);

        // constrain leaf_commit is set at all internal levels
        assert_array_eq(
            &mut builder.when(local.has_verifier_pvs),
            local.child_pvs.leaf_commit,
            leaf_commit,
        );

        // constrain verifier-specific pvs at all internal_recursive levels
        builder
            .when(local.child_pvs.internal_flag)
            .assert_zero(internal_flag.into() - AB::F::TWO);
        assert_array_eq(
            &mut builder.when(local.child_pvs.internal_flag),
            local.child_pvs.internal_for_leaf_commit,
            internal_for_leaf_commit,
        );

        // constrain verifier-specific pvs at internal_recursive levels after the first
        assert_array_eq(
            &mut builder
                .when(local.child_pvs.internal_flag * (local.child_pvs.internal_flag - AB::F::ONE)),
            local.child_pvs.internal_recursive_commit,
            internal_recursive_commit,
        );

        /*
         * We also need to receive cached commits from ProofShapeModule. The leaf verifier needs
         * to receive app_commit, and the internal verifier receives SymbolicExpressionAir's. Note
         * this interaction forces there to be exactly one cached trace per circuit.
         */
        let is_internal_flag_zero = (local.child_pvs.internal_flag - AB::F::ONE)
            * (local.child_pvs.internal_flag - AB::F::TWO)
            * AB::F::TWO.inverse();
        let is_internal_flag_one =
            -(local.child_pvs.internal_flag - AB::F::TWO) * local.child_pvs.internal_flag;
        let is_local_flag_two = (local.child_pvs.internal_flag - AB::F::ONE)
            * local.child_pvs.internal_flag
            * AB::F::TWO.inverse();
        let cached_commit = from_fn(|i| {
            is_leaf.clone() * local.child_pvs.program_commit[i]
                + is_internal
                    * (is_internal_flag_zero.clone() * local.child_pvs.leaf_commit[i]
                        + is_internal_flag_one.clone()
                            * local.child_pvs.internal_for_leaf_commit[i]
                        + is_local_flag_two.clone() * local.child_pvs.internal_recursive_commit[i])
        });

        self.cached_commit_bus.receive(
            builder,
            local.proof_idx,
            CachedCommitBusMessage {
                air_idx: select(
                    is_leaf.clone(),
                    AB::Expr::from_canonical_usize(PROGRAM_AIR_ID),
                    AB::Expr::from_canonical_usize(CONSTRAINT_EVAL_AIR_ID),
                ),
                cached_idx: select(
                    is_leaf.clone(),
                    AB::Expr::from_canonical_usize(PROGRAM_CACHED_TRACE_INDEX),
                    AB::Expr::from_canonical_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                ),
                cached_commit,
            },
            local.is_valid,
        );
    }
}
