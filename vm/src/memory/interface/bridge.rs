use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};

use crate::memory::interface::{
    air::MemoryInterfaceAir, columns::MemoryInterfaceCols, EXPAND_BUS, MEMORY_INTERFACE_BUS,
};

impl<const NUM_WORDS: usize, const WORD_SIZE: usize> MemoryInterfaceAir<NUM_WORDS, WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: MemoryInterfaceCols<NUM_WORDS, WORD_SIZE, AB::Var>,
    ) {
        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            (AB::Expr::one() - local.expand_direction) * AB::F::two().inverse(),
            AB::Expr::zero(),
            (local.address_space - AB::F::from_canonical_usize(self.memory_dimensions.as_offset))
                * AB::F::from_canonical_usize(1 << self.memory_dimensions.address_height),
            local.leaf_label.into(),
        ];
        expand_fields.extend(
            local
                .values
                .into_iter()
                .flat_map(|x| x.into_iter().map(Into::into)),
        );
        builder.push_send(EXPAND_BUS, expand_fields, local.expand_direction.into());

        for word_idx in 0..NUM_WORDS {
            let word = local.values[word_idx];

            // // when `expand_direction` is 1, `multiplicity` should be 2 * `auxes[i]`
            // // otherwise, `multiplicity` should be 2 * `expand_direction`
            // let multiplicity = local.expand_direction
            //     * (AB::Expr::two()
            //         - ((local.expand_direction + AB::F::one())
            //             * (AB::Expr::one() - local.auxes[word_idx])));

            builder.push_receive(
                MEMORY_INTERFACE_BUS,
                iter::once(local.address_space.into())
                    .chain(iter::once(
                        AB::Expr::from_canonical_usize(NUM_WORDS * WORD_SIZE) * local.leaf_label
                            + AB::F::from_canonical_usize(word_idx * WORD_SIZE),
                    ))
                    .chain(word.into_iter().map(Into::into))
                    .chain(iter::once(local.clks[word_idx].into())),
                // TODO[osama]: add timestamp to this interaction
                local.expand_direction,
            );
        }
    }
}
