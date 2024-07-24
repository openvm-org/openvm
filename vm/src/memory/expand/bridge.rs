use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};

use crate::memory::expand::{
    air::ExpandAir, columns::ExpandCols, EXPAND_BUS, POSEIDON2_DIRECT_REQUEST_BUS,
};

fn push_expand_send<const CHUNK: usize, AB: InteractionBuilder>(
    builder: &mut AB,
    sends: impl Into<AB::Expr>,
    is_final: impl Into<AB::Expr>,
    height: impl Into<AB::Expr>,
    label: impl Into<AB::Expr>,
    address_space: impl Into<AB::Expr>,
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
        let child_height = local.parent_height - AB::F::one();
        let two_inv = AB::F::two().inverse();

        push_expand_send(
            builder,
            -local.direction.into(),
            AB::Expr::from(two_inv) - local.direction * two_inv,
            local.parent_height,
            local.parent_label,
            local.address_space,
            local.parent_hash,
        );
        push_expand_send(
            builder,
            local.direction,
            AB::Expr::from(two_inv) - local.direction * two_inv + local.left_is_final,
            child_height.clone(),
            local.parent_label * AB::F::two(),
            local.address_space,
            local.left_child_hash,
        );
        push_expand_send(
            builder,
            local.direction,
            AB::Expr::from(two_inv) - local.direction * two_inv + local.right_is_final,
            child_height,
            local.parent_label * AB::F::two() + AB::F::one(),
            local.address_space,
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
