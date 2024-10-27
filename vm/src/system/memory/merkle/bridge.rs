use std::iter;

use ax_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::{
    arch::POSEIDON2_DIRECT_BUS,
    system::memory::merkle::{MemoryMerkleAir, MemoryMerkleCols},
};

#[derive(Copy, Clone, Debug)]
pub struct MemoryMerkleBus(pub usize);

impl<const CHUNK: usize> MemoryMerkleAir<CHUNK> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryMerkleCols<AB::Var, CHUNK>,
    ) {
        // interaction does not occur for first two rows;
        // for those, parent hash value comes from public values
        builder.push_send(
            self.merkle_bus.0,
            [
                local.expand_direction.into(),
                local.parent_height.into(),
                local.parent_as_label.into(),
                local.parent_address_label.into(),
            ]
            .into_iter()
            .chain(local.parent_hash.into_iter().map(Into::into)),
            // count can probably be made degree 1 if necessary
            (AB::Expr::one() - local.is_root) * local.expand_direction,
        );

        builder.push_receive(
            self.merkle_bus.0,
            [
                local.expand_direction + (local.left_direction_different * AB::F::two()),
                local.parent_height - AB::F::one(),
                local.parent_as_label * (AB::Expr::one() + local.height_section),
                local.parent_address_label * (AB::Expr::two() - local.height_section),
            ]
            .into_iter()
            .chain(local.left_child_hash.into_iter().map(Into::into)),
            local.expand_direction.into(),
        );

        builder.push_receive(
            self.merkle_bus.0,
            [
                local.expand_direction + (local.right_direction_different * AB::F::two()),
                local.parent_height - AB::F::one(),
                (local.parent_as_label * (AB::Expr::one() + local.height_section))
                    + local.height_section,
                (local.parent_address_label * (AB::Expr::two() - local.height_section))
                    + (AB::Expr::one() - local.height_section),
            ]
            .into_iter()
            .chain(local.right_child_hash.into_iter().map(Into::into)),
            local.expand_direction.into(),
        );

        let hash_fields = iter::empty()
            .chain(local.left_child_hash)
            .chain(local.right_child_hash)
            .chain(local.parent_hash);
        // TODO: do not hardcode the hash bus
        builder.push_send(
            POSEIDON2_DIRECT_BUS,
            hash_fields,
            local.expand_direction * local.expand_direction,
        );
    }
}
