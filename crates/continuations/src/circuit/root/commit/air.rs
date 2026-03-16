use std::borrow::Borrow;

use itertools::Itertools;
use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, SubAir};
use openvm_recursion_circuit::{bus::Poseidon2CompressBus, utils::assert_zeros};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

pub use crate::circuit::subair::UserPvsCommitCols;
use crate::circuit::{
    root::bus::{UserPvsCommitBus, UserPvsCommitTreeBus},
    subair::UserPvsCommitSubAir,
};

pub(super) const MAX_ENCODER_DEGREE: u32 = 3;

/**
 * Builds a binary Merkle tree to decommit and expose or emit the raw user public values.
 * Constrains that:
 * - leaf nodes read single digests from encoder-selected exposed public values, compress them
 *   with zeros, and compute leaf hashes
 * - internal nodes receive children from an internal permutation bus
 * - root commitment is sent to `UserPvsCommitBus`
 */
pub struct UserPvsCommitAir {
    pub subair: UserPvsCommitSubAir,
    encoder: Encoder,
    num_user_pvs: usize,
}

impl UserPvsCommitAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        user_pvs_commit_tree_bus: UserPvsCommitTreeBus,
        num_user_pvs: usize,
    ) -> Self {
        // Each leaf consumes `DIGEST_SIZE` public values, which are compressed with zeros
        // to compute the leaf hash. We require at least one leaf, and a full binary tree.
        debug_assert!(num_user_pvs >= DIGEST_SIZE);
        debug_assert!(num_user_pvs.is_multiple_of(DIGEST_SIZE));
        debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());
        let encoder = Encoder::new(num_user_pvs / DIGEST_SIZE, MAX_ENCODER_DEGREE, true);

        UserPvsCommitAir {
            subair: UserPvsCommitSubAir::new(
                poseidon2_compress_bus,
                user_pvs_commit_bus,
                user_pvs_commit_tree_bus,
            ),
            encoder,
            num_user_pvs,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitAir {
    fn width(&self) -> usize {
        UserPvsCommitCols::<u8>::width() + self.encoder.width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsCommitAir {
    fn num_public_values(&self) -> usize {
        self.num_user_pvs
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
        self.subair.eval(builder, (local, next, num_rows.into()));

        /*
         * Constrain that the left_child of each leaf node at row_idx corresponds to this
         * AIR's public values. Leaf nodes correspond to the raw user public values, and
         * leaf rows should be in order of their position in the public values vector. A
         * row is a leaf node if its receive_type == 1.
         */
        let is_leaf = local.receive_type * (AB::Expr::TWO - local.receive_type);
        assert_zeros(&mut builder.when(is_leaf.clone()), local.right_child);

        debug_assert_eq!(self.encoder.width(), row_idx_flags.len());
        self.encoder.eval(builder, row_idx_flags);
        builder.assert_eq(self.encoder.is_valid::<AB>(row_idx_flags), is_leaf.clone());

        let pvs = builder.public_values().iter().copied().collect_vec();
        let mut pvs_digest = [AB::Expr::ZERO; DIGEST_SIZE];
        for (pv_chunk_idx, pvs_chunk) in pvs.chunks(DIGEST_SIZE).enumerate() {
            let selected = self
                .encoder
                .get_flag_expr::<AB>(pv_chunk_idx, row_idx_flags);
            builder
                .when(selected.clone())
                .assert_eq(AB::Expr::from_usize(pv_chunk_idx), local.row_idx);
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
}
