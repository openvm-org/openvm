use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};

use crate::memory::interface::{
    air::MemoryInterfaceAir, columns::MemoryInterfaceCols, EXPAND_BUS, MEMORY_INTERFACE_BUS,
};

impl<const CHUNK: usize> MemoryInterfaceAir<CHUNK> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: MemoryInterfaceCols<CHUNK, AB::Var>,
    ) {
        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            (AB::Expr::one() - local.expand_direction) * AB::F::two().inverse(),
            local.address_space.into(),
            AB::Expr::zero(),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.values.map(AB::Var::into));
        builder.push_send(EXPAND_BUS, expand_fields, local.expand_direction.into());

        for i in 0..CHUNK {
            // when `expand_direction` is  1, `origin_expand_direction` should be 1
            // when `expand_direction` is -1, `origin_expand_direction` should be (2 * `auxes[i]`) - 1
            let origin_expand_direction = local.expand_direction
                + ((AB::Expr::one() - local.expand_direction) * local.auxes[i]);

            // when `expand_direction` is 1, `multiplicity` should be 2 * `auxes[i]`
            // otherwise, `multiplicity` should be 2 * `expand_direction`
            let multiplicity = local.expand_direction
                * (AB::Expr::two()
                    - ((local.expand_direction + AB::F::one())
                        * (AB::Expr::one() - local.auxes[i])));

            builder.push_receive(
                MEMORY_INTERFACE_BUS,
                [
                    origin_expand_direction,
                    local.address_space.into(),
                    (AB::Expr::from_canonical_usize(CHUNK) * local.leaf_label)
                        + AB::F::from_canonical_usize(i),
                    local.values[i].into(),
                ],
                multiplicity,
            );
        }
    }
}
