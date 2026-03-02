use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::{encoder::Encoder, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use recursion_circuit::{bus::Poseidon2CompressBus, utils::assert_zeros};

pub use crate::circuit::subair::UserPvsCommitCols;
use crate::circuit::{
    deferral::verify::{
        bus::{OutputValBus, OutputValMessage},
        output::VALS_IN_DIGEST,
    },
    root::bus::{UserPvsCommitBus, UserPvsCommitTreeBus},
    subair::UserPvsCommitSubAir,
};

#[derive(Clone, Debug)]
pub enum UserPvsCommitMode {
    ExposePublicValues { encoder: Encoder },
    SendToOutputValBus { output_val_bus: OutputValBus },
}

impl UserPvsCommitMode {
    pub fn width(&self) -> usize {
        match self {
            UserPvsCommitMode::ExposePublicValues { encoder } => encoder.width(),
            UserPvsCommitMode::SendToOutputValBus { .. } => 0,
        }
    }

    pub const fn exposes_public_values(&self) -> bool {
        matches!(self, Self::ExposePublicValues { .. })
    }
}

/**
 * Builds a binary Merkle tree to decommit and expose or emit the raw user public values.
 * Constrains that:
 * - leaf nodes read single digests, compress with zeros, and compute leaf hashes
 * - internal nodes receive children from an internal permutation bus
 * - root commitment is sent to `UserPvsCommitBus`
 * - leaf payload is either read from exposed public values (encoder-selected) or sent on
 *   `OutputValBus` starting at OUTPUT_USER_PVS_START_IDX
 */
pub struct UserPvsCommitAir {
    pub subair: UserPvsCommitSubAir,
    pub output_val_bus: OutputValBus,
    num_user_pvs: usize,
}

impl UserPvsCommitAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        user_pvs_commit_tree_bus: UserPvsCommitTreeBus,
        output_val_bus: OutputValBus,
        num_user_pvs: usize,
    ) -> Self {
        // Each leaf consumes `DIGEST_SIZE` public values, which are compressed with zeros
        // to compute the leaf hash. We require at least one leaf, and a full binary tree.
        debug_assert!(num_user_pvs >= DIGEST_SIZE);
        debug_assert!(num_user_pvs.is_multiple_of(DIGEST_SIZE));
        debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

        UserPvsCommitAir {
            subair: UserPvsCommitSubAir::new(
                poseidon2_compress_bus,
                user_pvs_commit_bus,
                user_pvs_commit_tree_bus,
            ),
            output_val_bus,
            num_user_pvs,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitAir {
    fn width(&self) -> usize {
        UserPvsCommitCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsCommitAir {}
impl<F> PartitionedBaseAir<F> for UserPvsCommitAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for UserPvsCommitAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let const_width = UserPvsCommitCols::<u8>::width();
        let row_idx_flags = &(*local)[const_width..];

        let local: &UserPvsCommitCols<AB::Var> = (*local)[..const_width].borrow();
        let next: &UserPvsCommitCols<AB::Var> = (*next)[..const_width].borrow();

        let num_rows = AB::F::from_usize(2 * self.num_user_pvs / DIGEST_SIZE);
        self.subair.eval(builder, (local, next, num_rows.into()));

        /*
         * Send the left_child of each leaf node to output_values to be processed
         * elsewhere. Note that in this case, this AIR has no public values. Also,
         * output_val_bus expects to receive the app_exe_commit and app_vk_commit
         * at indices 0..OUTPUT_USER_PVS_START_IDX. Note a row is a leaf node if
         * its receive_type == 1.
         */
        let is_leaf = local.receive_type * (AB::Expr::TWO - local.receive_type);
        assert_zeros(&mut builder.when(is_leaf.clone()), local.right_child);

        const OUTPUT_USER_PVS_START_IDX: usize = (2 * DIGEST_SIZE) / VALS_IN_DIGEST;
        const OUTPUT_VAL_MSGS_PER_ROW: usize = DIGEST_SIZE / VALS_IN_DIGEST;

        debug_assert_eq!(row_idx_flags.len(), 0);

        for (i, output_values) in local.left_child.chunks_exact(VALS_IN_DIGEST).enumerate() {
            self.output_val_bus.send(
                builder,
                OutputValMessage {
                    values: from_fn(|i| output_values[i].into()),
                    idx: AB::Expr::from_usize(OUTPUT_USER_PVS_START_IDX + i)
                        + local.row_idx * AB::Expr::from_usize(OUTPUT_VAL_MSGS_PER_ROW),
                },
                is_leaf.clone(),
            );
        }
    }
}
