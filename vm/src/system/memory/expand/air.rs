use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::system::memory::expand::{columns::MemoryMerkleCols, MemoryDimensions, MemoryMerkleBus};

#[derive(Clone, Debug)]
pub struct MemoryMerkleAir<const CHUNK: usize> {
    pub memory_dimensions: MemoryDimensions,
    pub merkle_bus: MemoryMerkleBus,
}

impl<const CHUNK: usize, F: Field> PartitionedBaseAir<F> for MemoryMerkleAir<CHUNK> {}
impl<const CHUNK: usize, F: Field> BaseAir<F> for MemoryMerkleAir<CHUNK> {
    fn width(&self) -> usize {
        MemoryMerkleCols::<CHUNK, F>::get_width()
    }
}
impl<const CHUNK: usize, F: Field> BaseAirWithPublicValues<F> for MemoryMerkleAir<CHUNK> {
    fn num_public_values(&self) -> usize {
        2 * CHUNK
    }
}

impl<const CHUNK: usize, AB: InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for MemoryMerkleAir<CHUNK>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = MemoryMerkleCols::<CHUNK, AB::Var>::from_slice(&local);
        let next = main.row_slice(1);
        let next = MemoryMerkleCols::<CHUNK, AB::Var>::from_slice(&next);

        // `expand_direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        builder.assert_bool(local.left_direction_different);
        builder.assert_bool(local.right_direction_different);

        // if `expand_direction` != -1, then `*_direction_different` should be 0
        builder
            .when_ne(local.expand_direction, AB::F::neg_one())
            .assert_zero(local.left_direction_different);
        builder
            .when_ne(local.expand_direction, AB::F::neg_one())
            .assert_zero(local.right_direction_different);

        // rows should be sorted in descending order
        // independently by `parent_height`, `height_section`, `is_root`
        builder
            .when_transition()
            .assert_bool(local.parent_height - next.parent_height);
        builder
            .when_transition()
            .assert_bool(local.height_section - next.height_section);
        builder
            .when_transition()
            .assert_bool(local.is_root - next.is_root);

        // row with greatest height should have `height_section` = 1
        builder.when_first_row().assert_one(local.height_section);
        // two rows with greatest height should have `is_root` = 1
        builder.when_first_row().assert_one(local.is_root);
        builder.when_first_row().assert_one(next.is_root);
        // row with least height should have `height_section` = 0, `is_root` = 0
        builder.when_last_row().assert_zero(local.height_section);
        builder.when_last_row().assert_zero(local.is_root);
        // `height_section` changes from 0 to 1 only when `parent_height` changes from `address_height` to `address_height` + 1
        builder
            .when_transition()
            .when_ne(
                local.parent_height,
                AB::F::from_canonical_usize(self.memory_dimensions.address_height + 1),
            )
            .assert_eq(local.height_section, next.height_section);
        builder
            .when_transition()
            .when_ne(
                next.parent_height,
                AB::F::from_canonical_usize(self.memory_dimensions.address_height),
            )
            .assert_eq(local.height_section, next.height_section);
        // two adjacent rows with `is_root` = 1 should have
        // the first `expand_direction` = 1, the second `expand_direction` = -1
        builder
            .when(local.is_root)
            .when(next.is_root)
            .assert_eq(local.expand_direction - next.expand_direction, AB::F::two());

        // roots should have correct height
        builder.when(local.is_root).assert_eq(
            local.parent_height,
            AB::Expr::from_canonical_usize(self.memory_dimensions.overall_height()),
        );

        // constrain public values
        for i in 0..CHUNK {
            let initial_hash_elem = builder.public_values()[i];
            let final_hash_elem = builder.public_values()[CHUNK + i];
            builder
                .when_first_row()
                .assert_eq(local.parent_hash[i], initial_hash_elem);
            builder
                .when_first_row()
                .assert_eq(next.parent_hash[i], final_hash_elem);
        }

        self.eval_interactions(builder, local);
    }
}
