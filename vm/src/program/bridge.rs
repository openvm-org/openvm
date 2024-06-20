use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use crate::cpu::READ_INSTRUCTION_BUS;

use super::{columns::ProgramPreprocessedCols, ProgramAir};

impl<F: PrimeField64> AirBridge<F> for ProgramAir<F> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let width = ProgramPreprocessedCols::<F>::get_width();

        let interactions = vec![Interaction {
            fields: (0..width)
                .map(|col| VirtualPairCol::single_preprocessed(col))
                .collect(),
            count: VirtualPairCol::single_main(0),
            argument_index: READ_INSTRUCTION_BUS,
        }];

        interactions
    }
}
