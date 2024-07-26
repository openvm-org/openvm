use std::iter;

use p3_field::{AbstractField, Field};

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
        // direction =  1   => parent_is_final = 0
        // direction = -1   => parent_is_final = 1
        let parent_is_final = (AB::Expr::one() - local.direction) * AB::F::two().inverse();

        builder.push_send(
            EXPAND_BUS,
            [
                parent_is_final.clone(),
                local.address_space.into(),
                local.parent_height.into(),
                local.parent_label.into(),
            ]
            .into_iter()
            .chain(local.parent_hash.into_iter().map(Into::into)),
            local.direction.into(),
        );

        builder.push_receive(
            EXPAND_BUS,
            [
                parent_is_final.clone() + local.left_is_final,
                local.address_space.into(),
                local.parent_height - AB::F::one(),
                local.parent_label * AB::F::two(),
            ]
            .into_iter()
            .chain(local.left_child_hash.into_iter().map(Into::into)),
            local.direction.into(),
        );

        builder.push_receive(
            EXPAND_BUS,
            [
                parent_is_final.clone() + local.right_is_final,
                local.address_space.into(),
                local.parent_height - AB::F::one(),
                (local.parent_label * AB::F::two()) + AB::F::one(),
            ]
            .into_iter()
            .chain(local.right_child_hash.into_iter().map(Into::into)),
            local.direction.into(),
        );

        let hash_fields = iter::empty()
            .chain(local.left_child_hash)
            .chain(local.right_child_hash)
            .chain(local.parent_hash);
        // TODO: do not hardcode the hash bus
        builder.push_send(POSEIDON2_DIRECT_REQUEST_BUS, hash_fields, AB::F::one());
    }
}
