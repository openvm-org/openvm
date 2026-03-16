use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
        PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::DIGEST_SIZE,
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::assert_zeros,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    circuit::deferral::{
        inner::bus::{InputOrMerkleCommitBus, InputOrMerkleCommitMessage},
        DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct InputCommitCols<F> {
    pub is_valid: F,
    pub is_first: F,

    pub proof_idx: F,
    pub has_verifier_pvs: F,

    pub air_idx: F,
    pub cached_idx: F,
    pub current_commit: [F; DIGEST_SIZE],
    pub capacity: [F; DIGEST_SIZE],

    pub res_left: [F; DIGEST_SIZE],
    pub res_right: [F; DIGEST_SIZE],
}

pub struct InputCommitAir {
    pub public_values_bus: PublicValuesBus,
    pub poseidon2_bus: Poseidon2PermuteBus,
    pub cached_commit_bus: CachedCommitBus,
    pub input_or_merkle_commit_bus: InputOrMerkleCommitBus,
}

impl<F> BaseAir<F> for InputCommitAir {
    fn width(&self) -> usize {
        InputCommitCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for InputCommitAir {}
impl<F> PartitionedBaseAir<F> for InputCommitAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for InputCommitAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &InputCommitCols<AB::Var> = (*local).borrow();
        let next: &InputCommitCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1> {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid,
                    counter: [local.proof_idx],
                    is_first: [local.is_first],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx],
                    is_first: [next.is_first],
                }
                .map_into(),
            ),
        );

        /*
         * Constrain that if has_verifier_pvs is consistent over all valid rows.
         */
        builder.assert_bool(local.has_verifier_pvs);
        builder
            .when(local.is_valid * next.is_valid)
            .assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);
        builder
            .when(local.is_valid * local.has_verifier_pvs)
            .assert_one(local.is_first);

        /*
         * Read the input commit from public values on the first row for each proof. This is the
         * input commit for the leaf verifier, and the deferral aggregation Merkle commit at all
         * internal levels.
         */
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;
        let air_idx = is_leaf.clone() * AB::Expr::from_usize(DEF_CIRCUIT_PVS_AIR_ID)
            + is_internal * AB::Expr::from_usize(DEF_AGG_PVS_AIR_ID);

        for (pv_idx, value) in local.current_commit.iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: air_idx.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                local.is_first,
            );
        }

        /*
         * Receive cached trace commits and fold them into the sponge on all non-first rows.
         */
        self.cached_commit_bus.receive(
            builder,
            local.proof_idx,
            CachedCommitBusMessage {
                air_idx: local.air_idx,
                cached_idx: local.cached_idx,
                cached_commit: local.current_commit,
            },
            is_leaf.clone() * (local.is_valid - local.is_first),
        );

        /*
         * Fold the received current_commit into the sponge. On the first row, the capacity
         * should be all zeros. On subsequent rows, the capacity should match the res_right
         * from the previous row.
         */
        let is_transition = next.is_valid - next.is_first;

        assert_zeros(&mut builder.when(local.is_first), local.capacity);
        assert_array_eq(
            &mut builder.when(is_leaf.clone() * is_transition.clone()),
            local.res_right,
            next.capacity,
        );

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: digests_to_poseidon2_input(local.current_commit, local.capacity),
                output: digests_to_poseidon2_input(local.res_left, local.res_right),
            },
            local.is_valid * is_leaf.clone(),
        );

        /*
         * Finally, on the last row for this proof we send the input commit.
         */
        self.input_or_merkle_commit_bus.send(
            builder,
            local.proof_idx,
            InputOrMerkleCommitMessage {
                has_verifier_pvs: local.has_verifier_pvs.into(),
                commit: from_fn(|i| {
                    is_internal * local.current_commit[i] + is_leaf.clone() * local.res_left[i]
                }),
            },
            local.is_valid * not::<AB::Expr>(is_transition),
        );
    }
}
