use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::PairBuilder;
use p3_field::Field;
use p3_matrix::Matrix;

use crate::cpu::READ_INSTRUCTION_BUS;

use super::ProgramAir;

impl<F: Field> ProgramAir<F> {
    pub fn eval_interactions<AB: PairBuilder<F = F> + InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let execution_frequency = main.row_slice(0)[0];
        let preprocessed = &builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let fields = prep_local.iter().map(|&x| x.into());

        builder.push_receive(READ_INSTRUCTION_BUS, fields, execution_frequency);
    }
}
