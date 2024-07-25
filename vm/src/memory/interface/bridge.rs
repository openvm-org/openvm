use p3_field::{AbstractField, Field};

use afs_stark_backend::interaction::InteractionBuilder;

use crate::memory::interface::air::MemoryInterfaceAir;
use crate::memory::interface::columns::MemoryInterfaceCols;
use crate::memory::interface::{EXPAND_BUS, MEMORY_INTERFACE_BUS};

impl<const CHUNK: usize> MemoryInterfaceAir<CHUNK> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: MemoryInterfaceCols<CHUNK, AB::Var>,
    ) {
        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            (AB::Expr::one() - local.direction) * AB::F::two().inverse(),
            local.address_space.into(),
            AB::Expr::zero(),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.values.map(AB::Var::into));
        builder.push_receive(EXPAND_BUS, expand_fields, local.direction.into());

        for i in 0..CHUNK {
            // when `direction` is -1, `is_final` should be `auxes[i]`
            // otherwise, `is_final` field should be 0
            let is_final =
                (AB::Expr::one() - local.direction) * AB::F::two().inverse() * local.auxes[i];

            // when `direction` is 1, `multiplicity` should be `auxes[i]`
            // otherwise, `multiplicity` should be `direction`
            let multiplicity = local.direction
                * (AB::Expr::one()
                    - ((local.direction + AB::F::one())
                        * AB::F::two().inverse()
                        * (AB::Expr::one() - local.auxes[i])));

            builder.push_send(
                MEMORY_INTERFACE_BUS,
                [
                    is_final,
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
