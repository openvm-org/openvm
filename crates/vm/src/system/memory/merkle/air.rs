use std::{borrow::Borrow, iter};

use openvm_circuit_primitives::ColumnsAir;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::system::memory::merkle::{MemoryDimensions, MemoryMerkleCols, MemoryMerklePvs};

#[derive(Clone, Debug, ColumnsAir)]
#[columns_via(MemoryMerkleCols<u8, DIGEST_WIDTH>)]
pub struct MemoryMerkleAir<const DIGEST_WIDTH: usize> {
    pub memory_dimensions: MemoryDimensions,
    pub merkle_bus: PermutationCheckBus,
    pub compression_bus: PermutationCheckBus,
}

impl<const DIGEST_WIDTH: usize, F: Field> PartitionedBaseAir<F> for MemoryMerkleAir<DIGEST_WIDTH> {}
impl<const DIGEST_WIDTH: usize, F: Field> BaseAir<F> for MemoryMerkleAir<DIGEST_WIDTH> {
    fn width(&self) -> usize {
        MemoryMerkleCols::<F, DIGEST_WIDTH>::width()
    }
}
impl<const DIGEST_WIDTH: usize, F: Field> BaseAirWithPublicValues<F>
    for MemoryMerkleAir<DIGEST_WIDTH>
{
    fn num_public_values(&self) -> usize {
        MemoryMerklePvs::<F, DIGEST_WIDTH>::width()
    }
}

impl<const DIGEST_WIDTH: usize, AB: InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for MemoryMerkleAir<DIGEST_WIDTH>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &MemoryMerkleCols<_, DIGEST_WIDTH> = (*local).borrow();
        let next: &MemoryMerkleCols<_, DIGEST_WIDTH> = (*next).borrow();

        // `expand_direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        // the following set of constrains enforces left/right_child_mode value depending on
        // expand_direction
        for m in [local.left_child_mode, local.right_child_mode] {
            // mode must be in 0, 1, 2
            builder.assert_zero(m * (AB::Expr::ONE - m) * (AB::Expr::TWO - m));
            // mode must not be 2 when expand_direction=-1 or expand_direction=0
            builder
                .when_ne(local.expand_direction, AB::Expr::ONE)
                .assert_zero(m * (AB::Expr::ONE - m));
            // if mode is not 0, then expand direction is -1 or 1
            builder
                .when(m)
                .assert_zero(AB::Expr::ONE - local.expand_direction * local.expand_direction);
        }

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
        // the root rows have `as_label` and `address_label` set to zero,
        // so that we can't use another tree representation
        builder
            .when(local.is_root)
            .assert_zero(local.parent_address_label);
        builder
            .when(local.is_root)
            .assert_zero(local.parent_as_label);
        // row with least height should have `height_section` = 0, `is_root` = 0
        builder.when_last_row().assert_zero(local.height_section);
        builder.when_last_row().assert_zero(local.is_root);
        // `height_section` changes from 0 to 1 only when `parent_height` changes from
        // `address_height` to `address_height` + 1
        builder
            .when_transition()
            .when_ne(
                local.parent_height,
                AB::F::from_usize(self.memory_dimensions.address_height + 1),
            )
            .assert_eq(local.height_section, next.height_section);
        builder
            .when_transition()
            .when_ne(
                next.parent_height,
                AB::F::from_usize(self.memory_dimensions.address_height),
            )
            .assert_eq(local.height_section, next.height_section);
        // two adjacent rows with `is_root` = 1 should have
        // the first `expand_direction` = 1, the second `expand_direction` = -1
        builder
            .when(local.is_root)
            .when(next.is_root)
            .assert_eq(local.expand_direction - next.expand_direction, AB::F::TWO);

        // roots should have correct height
        builder.when(local.is_root).assert_eq(
            local.parent_height,
            AB::Expr::from_usize(self.memory_dimensions.overall_height()),
        );

        // parent height should not be zero when `expand_direction` != 0
        builder
            .when_ne(local.expand_direction, AB::F::ZERO)
            .assert_eq(local.parent_height * local.parent_height_inv, AB::F::ONE);

        // constrain public values
        let &MemoryMerklePvs::<_, DIGEST_WIDTH> {
            initial_root,
            final_root,
        } = builder.public_values().borrow();
        for i in 0..DIGEST_WIDTH {
            builder
                .when_first_row()
                .assert_eq(local.parent_hash[i], initial_root[i]);
            builder
                .when_first_row()
                .assert_eq(next.parent_hash[i], final_root[i]);
        }

        self.eval_interactions(builder, local);
    }
}

impl<const DIGEST_WIDTH: usize> MemoryMerkleAir<DIGEST_WIDTH> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryMerkleCols<AB::Var, DIGEST_WIDTH>,
    ) {
        // interaction does not occur for first two rows;
        // for those, parent hash value comes from public values
        self.merkle_bus.interact(
            builder,
            [
                local.expand_direction.into(),
                local.parent_height.into(),
                local.parent_as_label.into(),
                local.parent_address_label.into(),
            ]
            .into_iter()
            .chain(local.parent_hash.into_iter().map(Into::into)),
            // count can probably be made degree 1 if necessary
            (AB::Expr::ONE - local.is_root) * local.expand_direction,
        );

        // The child interactions reuse one `*_child_mode` column per side (see
        // `MemoryMerkleCols`). Writing `dir = expand_direction` and `m = *_child_mode`:
        //
        //   tag   = dir + m*(1 - dir)
        //             dir= 1 (initial):  1              (m does not affect the tag)
        //             dir=-1 (final):    2*m - 1        (m is the dd bit: 0 -> -1, 1 -> +1)
        //             dir= 0 (padding):  0              (m is forced 0)
        //   count = (dir*(dir - 1) - m*(1 + dir)) / 2
        //             dir= 1 (initial): -m              (consume the child's initial claim m times)
        //             dir=-1 (final):   +1              (send; independent of m)
        //             dir= 0 (padding):  0
        let one_half = AB::F::TWO.inverse();

        let left_tag = local.expand_direction
            + local.left_child_mode * (AB::Expr::ONE - local.expand_direction);
        let left_count = (local.expand_direction * (local.expand_direction - AB::Expr::ONE)
            - local.left_child_mode * (AB::Expr::ONE + local.expand_direction))
            * AB::Expr::from(one_half);
        self.merkle_bus.interact(
            builder,
            [
                left_tag,
                local.parent_height - AB::F::ONE,
                local.parent_as_label * (AB::Expr::ONE + local.height_section),
                local.parent_address_label * (AB::Expr::TWO - local.height_section),
            ]
            .into_iter()
            .chain(local.left_child_hash.into_iter().map(Into::into)),
            left_count,
        );

        let right_tag = local.expand_direction
            + local.right_child_mode * (AB::Expr::ONE - local.expand_direction);
        let right_count = (local.expand_direction * (local.expand_direction - AB::Expr::ONE)
            - local.right_child_mode * (AB::Expr::ONE + local.expand_direction))
            * AB::Expr::from(one_half);
        self.merkle_bus.interact(
            builder,
            [
                right_tag,
                local.parent_height - AB::F::ONE,
                (local.parent_as_label * (AB::Expr::ONE + local.height_section))
                    + local.height_section,
                (local.parent_address_label * (AB::Expr::TWO - local.height_section))
                    + (AB::Expr::ONE - local.height_section),
            ]
            .into_iter()
            .chain(local.right_child_hash.into_iter().map(Into::into)),
            right_count,
        );

        let compress_fields = iter::empty()
            .chain(local.left_child_hash)
            .chain(local.right_child_hash)
            .chain(local.parent_hash);
        self.compression_bus.interact(
            builder,
            compress_fields,
            local.expand_direction * local.expand_direction,
        );
    }
}
