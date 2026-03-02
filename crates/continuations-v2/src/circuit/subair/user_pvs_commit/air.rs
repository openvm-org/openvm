use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::AirBuilder;
use p3_field::{Field, PrimeCharacteristicRing};
use recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    circuit::root::bus::{
        UserPvsCommitBus, UserPvsCommitMessage, UserPvsCommitTreeBus, UserPvsCommitTreeMessage,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsCommitCols<T> {
    pub row_idx: T,

    // 1 for internal, 2 for root, 0 for invalid
    pub send_type: T,
    // 1 for pvs, 2 for internal, 0 for invalid
    pub receive_type: T,

    pub parent: [T; DIGEST_SIZE],
    pub is_right_child: T,

    pub left_child: [T; DIGEST_SIZE],
    pub right_child: [T; DIGEST_SIZE],
}

/// Shared constraints for user public values commitment tree AIRs.
#[derive(Clone, Debug, derive_new::new)]
pub struct UserPvsCommitSubAir {
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub user_pvs_commit_tree_bus: UserPvsCommitTreeBus,
}

impl<AB: AirBuilder + InteractionBuilder> SubAir<AB> for UserPvsCommitSubAir {
    type AirContext<'a>
        = (
        &'a UserPvsCommitCols<AB::Var>,
        &'a UserPvsCommitCols<AB::Var>,
        AB::Expr,
    )
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let (local, next, num_rows) = ctx;

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
            .assert_eq(local.row_idx, num_rows.clone() - AB::F::ONE);

        /*
         * Constrain that is_right_child is correctly set. Even row_idx correspond to
         * left children, and odd row_idx correspond to right - thus we constrain that
         * the is_right_child flag alternates.
         */
        builder.when_first_row().assert_zero(local.is_right_child);
        builder.assert_eq(local.is_right_child, AB::Expr::ONE - next.is_right_child);

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

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.left_child, local.right_child),
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
    }
}
