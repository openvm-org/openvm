use afs_middleware::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::XorCols, XorChip};

impl<F: PrimeField64, const N: usize> Chip<F> for XorChip<N> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = XorCols::<N, F>::get_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();
        let cols_to_send = XorCols::<N, F>::cols_to_send(&all_cols);

        let mut fields = vec![];
        for col in cols_to_send {
            fields.push(VirtualPairCol::single_main(col));
        }

        vec![Interaction {
            fields: fields,
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
