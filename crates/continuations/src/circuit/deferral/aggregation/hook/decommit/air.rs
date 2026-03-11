use std::borrow::Borrow;

use openvm_circuit_primitives::{utils::not, SubAir};
use openvm_recursion_circuit::utils::assert_zeros;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::circuit::{
    deferral::aggregation::hook::bus::{IoCommitBus, IoCommitMessage},
    subair::{MerkleTreeCols, MerkleTreeSubAir},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleDecommitCols<F> {
    pub merkle_tree_cols: MerkleTreeCols<F>,
    pub send_commits: F,
    pub num_rows: F,
}

pub struct MerkleDecommitAir {
    pub subair: MerkleTreeSubAir,
    pub io_commit_bus: IoCommitBus,
}

impl<F> BaseAir<F> for MerkleDecommitAir {
    fn width(&self) -> usize {
        MerkleDecommitCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for MerkleDecommitAir {}
impl<F> PartitionedBaseAir<F> for MerkleDecommitAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for MerkleDecommitAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &MerkleDecommitCols<AB::Var> = (*local).borrow();
        let next: &MerkleDecommitCols<AB::Var> = (*next).borrow();

        self.subair.eval(
            builder,
            (
                &local.merkle_tree_cols,
                &next.merkle_tree_cols,
                local.num_rows.into(),
            ),
        );
        builder.assert_eq(local.num_rows, next.num_rows);

        /*
         * We send the input and output commits (i.e. left_child and right_child) to
         * OnionHashAir when send_commits is 1. We constrain that we send at least one
         * commit by forcing it to be 1 on the first row, and that all the sends are
         * at the beginning. Additionally, we need constrain that send_commits is only
         * 1 on Merkle leaves, and that non-send Merkle leaves are unset.
         */
        builder.assert_bool(local.send_commits);
        builder.when_first_row().assert_one(local.send_commits);
        builder
            .when_transition()
            .assert_bool(local.send_commits - next.send_commits);

        let is_leaf = local.merkle_tree_cols.receive_type
            * (AB::Expr::TWO - local.merkle_tree_cols.receive_type);
        builder.when(local.send_commits).assert_one(is_leaf.clone());

        let mut when_no_send = builder.when(not(local.send_commits));
        let mut when_no_send_and_leaf = when_no_send.when(is_leaf);
        assert_zeros(
            &mut when_no_send_and_leaf,
            local.merkle_tree_cols.left_child,
        );
        assert_zeros(
            &mut when_no_send_and_leaf,
            local.merkle_tree_cols.right_child,
        );

        self.io_commit_bus.send(
            builder,
            IoCommitMessage {
                idx: local.merkle_tree_cols.row_idx,
                input_commit: local.merkle_tree_cols.left_child,
                output_commit: local.merkle_tree_cols.right_child,
            },
            local.send_commits,
        );
    }
}
