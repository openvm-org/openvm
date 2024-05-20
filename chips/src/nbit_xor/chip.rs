use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::NBitXorChip;

impl<F: PrimeField64, const N: usize, const M: usize> Chip<F> for NBitXorChip<N, M> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_limbs = (N + M - 1) / M;

        let mut interactions = vec![];

        for i in 0..num_limbs {
            interactions.push(Interaction {
                fields: vec![
                    VirtualPairCol::single_main(3 + i + M),
                    VirtualPairCol::single_main(3 + i + 2 * M),
                    VirtualPairCol::single_main(3 + i + 3 * M),
                ],
                count: VirtualPairCol::constant(F::one()),
                argument_index: self.bus_index(),
            });
        }

        interactions
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(0),
                VirtualPairCol::single_main(1),
                VirtualPairCol::single_main(2),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
