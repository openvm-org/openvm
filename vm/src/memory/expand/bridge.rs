use std::iter;

use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::memory::expand::{
    air::ExpandAir, columns::ExpandCols, EXPAND_BUS, POSEIDON2_DIRECT_REQUEST_BUS,
};

impl<const CHUNK: usize> ExpandAir<CHUNK> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: ExpandCols<CHUNK, AB::Var>,
    ) {
        builder.push_send(
            EXPAND_BUS,
            [
                local.expand_direction.into(),
                local.address_space.into(),
                local.parent_height.into(),
                local.parent_label.into(),
            ]
            .into_iter()
            .chain(local.parent_hash.into_iter().map(Into::into)),
            local.expand_direction.into(),
        );

        builder.push_receive(
            EXPAND_BUS,
            [
                local.expand_direction + (local.left_direction_different * AB::F::two()),
                local.address_space.into(),
                local.parent_height - AB::F::one(),
                local.parent_label * AB::F::two(),
            ]
            .into_iter()
            .chain(local.left_child_hash.into_iter().map(Into::into)),
            local.expand_direction.into(),
        );

        builder.push_receive(
            EXPAND_BUS,
            [
                local.expand_direction + (local.right_direction_different * AB::F::two()),
                local.address_space.into(),
                local.parent_height - AB::F::one(),
                (local.parent_label * AB::F::two()) + AB::F::one(),
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
            POSEIDON2_DIRECT_REQUEST_BUS,
            hash_fields,
            local.expand_direction * local.expand_direction,
        );
    }
}
