use std::iter;

use p3_field::{AbstractField, Field};

use afs_stark_backend::interaction::InteractionBuilder;

use crate::memory::expand::{
    air::ExpandAir, columns::ExpandCols, EXPAND_BUS, POSEIDON2_DIRECT_REQUEST_BUS,
};

fn push_expand_send<const CHUNK: usize, AB: InteractionBuilder>(
    builder: &mut AB,
    sends: impl Into<AB::Expr>,
    is_final: impl Into<AB::Expr>,
    address_space: impl Into<AB::Expr>,
    label: impl Into<AB::Expr>,
    height: impl Into<AB::Expr>,
    hash: [impl Into<AB::Expr>; CHUNK],
) {
    let fields = [
        is_final.into(),
        address_space.into(),
        height.into(),
        label.into(),
    ]
    .into_iter()
    .chain(hash.into_iter().map(Into::into));
    builder.push_send(EXPAND_BUS, fields, sends);
}

impl<const CHUNK: usize> ExpandAir<CHUNK> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: ExpandCols<CHUNK, AB::Var>,
    ) {
        // direction =  1   => parent_is_final = 0
        // direction = -1   => parent_is_final = 1
        let parent_is_final = (AB::Expr::one() - local.direction) * AB::F::two().inverse();

        push_expand_send(
            builder,
            -local.direction.into(),
            parent_is_final.clone(),
            local.address_space,
            local.parent_label,
            local.parent_height,
            local.parent_hash,
        );
        push_expand_send(
            builder,
            local.direction,
            parent_is_final.clone() + local.left_is_final,
            local.address_space,
            local.parent_label * AB::F::two(),
            local.parent_height - AB::F::one(),
            local.left_child_hash,
        );
        push_expand_send(
            builder,
            local.direction,
            parent_is_final.clone() + local.right_is_final,
            local.address_space,
            (local.parent_label * AB::F::two()) + AB::F::one(),
            local.parent_height - AB::F::one(),
            local.right_child_hash,
        );

        let hash_fields = iter::empty()
            .chain(local.left_child_hash)
            .chain(local.right_child_hash)
            .chain(local.parent_hash);
        // TODO: do not hardcode the hash bus
        builder.push_send(POSEIDON2_DIRECT_REQUEST_BUS, hash_fields, AB::F::one());
    }
}
