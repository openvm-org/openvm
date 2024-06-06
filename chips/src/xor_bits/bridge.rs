use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::{Field, PrimeField64};

use crate::sub_chip::SubAirBridge;

use super::{columns::XorCols, XorBitsAir};

impl<F: PrimeField64, const N: usize> AirBridge<F> for XorBitsAir<N> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = XorCols::<N, F>::get_width();
        let indices = (0..num_cols).collect::<Vec<usize>>();
        let col_indices = XorCols::<N, usize>::from_slice(&indices);

        SubAirBridge::receives(self, col_indices)
    }
}

impl<F: Field, const N: usize> SubAirBridge<F> for XorBitsAir<N> {
    fn receives(&self, col_indices: XorCols<N, usize>) -> Vec<Interaction<F>> {
        let io_indices = col_indices.io;
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(io_indices.x),
                VirtualPairCol::single_main(io_indices.y),
                VirtualPairCol::single_main(io_indices.z),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index,
        }]
    }
}
