use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use crate::poseidon2::{columns::Poseidon2Cols, Poseidon2Air};

// Receives input and output columns in one interaction
impl<const WIDTH: usize, F: Field> AirBridge<F> for Poseidon2Air<WIDTH, F> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let index_map = Poseidon2Cols::<WIDTH, F>::index_map(self);
        let field_cols = index_map
            .input
            .collect::<Vec<_>>()
            .into_iter()
            .chain(index_map.output.collect::<Vec<_>>());

        vec![Interaction {
            fields: field_cols.map(VirtualPairCol::single_main).collect(),
            count: VirtualPairCol::one(),
            argument_index: self.bus_index,
        }]
    }
}
