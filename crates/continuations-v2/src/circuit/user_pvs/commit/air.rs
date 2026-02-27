use std::{array::from_fn, borrow::Borrow};

use itertools::Itertools;
use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::{
    deferral::verify::{
        bus::{OutputValBus, OutputValMessage},
        output::VALS_IN_DIGEST,
    },
    root::bus::{
        UserPvsCommitBus, UserPvsCommitMessage, UserPvsCommitTreeBus, UserPvsCommitTreeMessage,
    },
};

pub(super) const MAX_ENCODER_DEGREE: u32 = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsCommitCols<F> {
    pub row_idx: F,

    // 1 for internal, 2 for root, 0 for invalid
    pub send_type: F,
    // 1 for pvs, 2 for internal, 0 for invalid
    pub receive_type: F,

    pub parent: [F; DIGEST_SIZE],
    pub is_right_child: F,

    pub left_child: [F; DIGEST_SIZE],
    pub right_child: [F; DIGEST_SIZE],
}

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
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub user_pvs_commit_tree_bus: UserPvsCommitTreeBus,

    pub mode: UserPvsCommitMode,
    num_user_pvs: usize,
}

impl UserPvsCommitAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        user_pvs_commit_tree_bus: UserPvsCommitTreeBus,
        output_val_bus: Option<OutputValBus>,
        num_user_pvs: usize,
    ) -> Self {
        // Each leaf consumes `DIGEST_SIZE` public values, which are compressed with zeros
        // to compute the leaf hash. We require at least one leaf, and a full binary tree.
        debug_assert!(num_user_pvs >= DIGEST_SIZE);
        debug_assert!(num_user_pvs.is_multiple_of(DIGEST_SIZE));
        debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

        let mode = if let Some(output_val_bus) = output_val_bus {
            UserPvsCommitMode::SendToOutputValBus { output_val_bus }
        } else {
            // One selector per leaf PV chunk (each leaf consumes 1 digest).
            let encoder = Encoder::new(num_user_pvs / DIGEST_SIZE, MAX_ENCODER_DEGREE, true);
            UserPvsCommitMode::ExposePublicValues { encoder }
        };

        UserPvsCommitAir {
            poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            mode,
            num_user_pvs,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitAir {
    fn width(&self) -> usize {
        UserPvsCommitCols::<u8>::width() + self.mode.width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsCommitAir {
    fn num_public_values(&self) -> usize {
        if self.mode.exposes_public_values() {
            self.num_user_pvs
        } else {
            0
        }
    }
}
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

        /*
         * Constrain that row_idx actually corresponds to the matrix row index, and
         * that there are exactly num_rows rows.
         */
        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_eq(local.row_idx + AB::F::ONE, next.row_idx);
        builder
            .when_last_row()
            .assert_eq(local.row_idx, num_rows - AB::F::ONE);

        /*
         * Constrain that is_right_child is correctly set. Even row_idx correspond to
         * left children, and odd row_idx correspond to right - thus we constrain that
         * the is_right_child flag alternates.
         */
        builder.when_first_row().assert_zero(local.is_right_child);
        builder.assert_eq(local.is_right_child, not(next.is_right_child));

        /*
         * Constrain that the first num_rows - 2 rows send their parent internally,
         * the second last row sends the commit, and the last row sends nothing. There
         * must be at least 2 rows, which is enforced by constraining send_type to be
         * 1 or 2 on the first row and 0 on the last.
         */
        builder.assert_tern(local.send_type);

        builder
            .when_first_row()
            .assert_bool(local.send_type - AB::F::ONE);
        builder
            .when_transition()
            .assert_bool(local.send_type - AB::F::ONE);
        builder.when_last_row().assert_zero(local.send_type);

        builder
            .when_ne(local.send_type, AB::F::TWO)
            .assert_bool(next.send_type - AB::F::ONE);
        builder
            .when_ne(local.send_type, AB::F::ZERO)
            .when_ne(local.send_type, AB::F::ONE)
            .assert_zero(next.send_type);

        /*
         * Constrain that the first num_rows / 2 rows read their left and right values
         * from pvs, the next (num_rows / 2) - 1 rows receive their values internally,
         * and that the last row receives no values. We constrain this by asserting the
         * last row's receive_type == 0 and that elsewhere the column is monotonically
         * increasing.
         */
        builder.assert_tern(local.receive_type);

        builder.when_first_row().assert_one(local.receive_type);
        builder
            .when_transition()
            .assert_bool(local.receive_type - AB::F::ONE);
        builder.when_last_row().assert_zero(local.receive_type);

        builder
            .when_ne(local.receive_type, AB::F::ZERO)
            .when_ne(local.receive_type, AB::F::ONE)
            .assert_zero(next.receive_type * (next.receive_type - AB::F::TWO));

        /*
         * Send and receive internal messages. Given a non-root node at index i, its
         * parent is at index floor((i + num_rows) / 2). Note that we send the
         * internal message when local.send_type == 1, and receive left and right
         * when internal_receive == 2.
         */
        let half = AB::F::TWO.inverse();
        let internal_send = local.send_type * (AB::Expr::TWO - local.send_type);
        let internal_receive = local.receive_type * (local.receive_type - AB::F::ONE) * half;

        self.user_pvs_commit_tree_bus.send(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.parent.map(Into::into),
                is_right_child: local.is_right_child.into(),
                parent_idx: (local.row_idx - local.is_right_child + num_rows) * half,
            },
            internal_send,
        );

        self.user_pvs_commit_tree_bus.receive(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.left_child.map(Into::into),
                is_right_child: AB::Expr::ZERO,
                parent_idx: local.row_idx.into(),
            },
            internal_receive.clone(),
        );

        self.user_pvs_commit_tree_bus.receive(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.right_child.map(Into::into),
                is_right_child: AB::Expr::ONE,
                parent_idx: local.row_idx.into(),
            },
            internal_receive,
        );

        /*
         * Send and receive external messages. At every valid node we have to receive
         * a message from the Poseidon2 peripheral AIR to constrain that parent is the
         * compression hash of left_child and right_child. At the root node we send the
         * commit to UserPvsInMemoryAir, which constrains that it is in the correct
         * position in the final memory Merkle tree. Note a row is valid iff it is not
         * the last, i.e. iff send_type (or equivalently receive_type) is non-zero.
         */
        let is_valid = local.send_type * (AB::Expr::from_u8(3) - local.send_type) * half;
        let is_root = local.send_type * (local.send_type - AB::F::ONE) * half;

        let poseidon2_input: [AB::Var; POSEIDON2_WIDTH] = local
            .left_child
            .into_iter()
            .chain(local.right_child)
            .collect_array()
            .unwrap();

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon2_input,
                output: local.parent,
            },
            is_valid,
        );

        self.user_pvs_commit_bus.send(
            builder,
            UserPvsCommitMessage {
                user_pvs_commit: local.parent,
            },
            is_root,
        );

        /*
         * Constrain that the left_child of each leaf node depending on this AIR's mode.
         * Leaf nodes correspond to the raw user public values, and leaf rows should be
         * in order of their position in the public values vector. A row is a leaf node
         * if its receive_type == 1.
         */
        let is_leaf = local.receive_type * (AB::Expr::TWO - local.receive_type);
        assert_zeros(&mut builder.when(is_leaf.clone()), local.right_child);

        match &self.mode {
            UserPvsCommitMode::ExposePublicValues { encoder } => {
                /*
                 * Constrain that the left_child of each leaf node at row_idx corresponds
                 * to this AIR's public values.
                 */
                debug_assert_eq!(encoder.width(), row_idx_flags.len());
                encoder.eval(builder, row_idx_flags);
                builder.assert_eq(encoder.is_valid::<AB>(row_idx_flags), is_leaf.clone());

                let pvs = builder.public_values();
                let mut pvs_digest = [AB::Expr::ZERO; DIGEST_SIZE];
                for (pv_chunk_idx, pvs_chunk) in pvs.chunks(DIGEST_SIZE).enumerate() {
                    let selected = encoder.get_flag_expr::<AB>(pv_chunk_idx, row_idx_flags);
                    for digest_idx in 0..DIGEST_SIZE {
                        pvs_digest[digest_idx] += selected.clone() * pvs_chunk[digest_idx].into();
                    }
                }

                assert_array_eq(
                    builder,
                    pvs_digest,
                    local.left_child.map(|x| x * is_leaf.clone()),
                );
            }
            UserPvsCommitMode::SendToOutputValBus { output_val_bus } => {
                /*
                 * Send the left_child of each leaf node to output_values to be processed
                 * elsewhere. Note that in this case, this AIR has no public values. Also,
                 * output_val_bus expects to receive the app_exe_commit and app_vk_commit
                 * at indices 0..OUTPUT_USER_PVS_START_IDX.
                 */
                const OUTPUT_USER_PVS_START_IDX: usize = (2 * DIGEST_SIZE) / VALS_IN_DIGEST;
                const OUTPUT_VAL_MSGS_PER_ROW: usize = DIGEST_SIZE / VALS_IN_DIGEST;

                debug_assert_eq!(row_idx_flags.len(), 0);

                for (i, output_values) in local.left_child.chunks_exact(VALS_IN_DIGEST).enumerate()
                {
                    output_val_bus.send(
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
    }
}
