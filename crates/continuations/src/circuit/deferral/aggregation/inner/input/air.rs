use std::borrow::Borrow;

use openvm_circuit_primitives::utils::not;
use openvm_recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::DIGEST_SIZE,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    circuit::deferral::{
        aggregation::inner::bus::{InputOrMerkleCommitBus, InputOrMerkleCommitMessage},
        DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct InputCommitCols<F> {
    // 1 if current_commit and cached_commit need to be hashed, 2 if we can
    // send the current_commit, 0 if invalid
    pub state: F,
    pub is_first: F,

    pub proof_idx: F,
    pub has_verifier_pvs: F,
    pub current_commit: [F; DIGEST_SIZE],

    pub air_idx: F,
    pub cached_idx: F,
    pub cached_commit: [F; DIGEST_SIZE],
}

pub struct InputCommitAir {
    pub public_values_bus: PublicValuesBus,
    pub poseidon2_bus: Poseidon2CompressBus,
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

        /*
         * Constrain that state is correctly set. The first row must be valid (i.e. have state
         * be 1 or 2), and all valid rows should be at the beginning. Note that proof_idx is
         * implicitly constrained in DeferralAggPvsAir, which receives one InputOrMerkleCommitBus
         * message per proof.
         */
        builder.assert_tern(local.state);
        builder
            .when_first_row()
            .assert_bool(local.state - AB::Expr::ONE);
        builder
            .when_transition()
            .when_ne(local.state, AB::Expr::ONE)
            .when_ne(local.state, AB::Expr::TWO)
            .assert_zero(next.state);

        let is_compress = local.state * (AB::Expr::TWO - local.state);
        builder
            .when(is_compress.clone())
            .assert_bool(next.state - AB::Expr::ONE);
        builder
            .when(is_compress.clone())
            .assert_eq(local.proof_idx, next.proof_idx);

        /*
         * Constrain is_first to be 1 on the first row, and 1 every time a row switches to a new
         * proof_idx. Also constrain that if has_verifier_pvs is 1, all rows are send rows.
         */
        builder.assert_bool(local.is_first);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when_ne(local.proof_idx, next.proof_idx)
            .when(next.state)
            .assert_one(next.is_first);

        builder.assert_bool(local.has_verifier_pvs);
        builder.assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);
        builder
            .when(local.has_verifier_pvs)
            .assert_eq(local.state, AB::Expr::TWO);

        /*
         * Read the input commit from public values on the first row for each proof. This is the
         * input commit for the leaf verifier, and the deferral aggregation Merkle commit at all
         * internal levels.
         */
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;
        let air_idx = is_leaf * AB::Expr::from_usize(DEF_CIRCUIT_PVS_AIR_ID)
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
         * Receive cached trace commits and hash them with current_commit. We should only
         * do this if state is 1.
         */
        self.cached_commit_bus.receive(
            builder,
            local.proof_idx,
            CachedCommitBusMessage {
                air_idx: local.air_idx,
                cached_idx: local.cached_idx,
                cached_commit: local.cached_commit,
            },
            is_compress.clone(),
        );

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.current_commit, local.cached_commit),
                output: next.current_commit,
            },
            is_compress,
        );

        /*
         * Finally, when state is 2 we want to send current_commit.
         */
        self.input_or_merkle_commit_bus.send(
            builder,
            local.proof_idx,
            InputOrMerkleCommitMessage {
                has_verifier_pvs: local.has_verifier_pvs,
                commit: local.current_commit,
            },
            local.state * (local.state - AB::Expr::ONE) * AB::F::TWO.inverse(),
        );
    }
}
