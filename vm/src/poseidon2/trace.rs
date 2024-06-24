use super::Poseidon2Air;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

impl<const WIDTH: usize> Poseidon2Air<WIDTH> {
    pub fn generate_trace(&self, input_states: Vec<Vec<BabyBear>>) -> RowMajorMatrix<BabyBear> {
        RowMajorMatrix::new(
            input_states.iter().map(|input_state| self.generate_local_trace(input_state.clone())).collect(),
            self::get_width()
        )
    }

    pub fn generate_local_trace(&self, input_state: Vec<BabyBear>) -> Vec<BabyBear> {
        let poseidon2 = Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, WIDTH, 7>::new(
            &self.rounds_f,
            &self.external_constants,
            Poseidon2ExternalMatrixGeneral,
            &self.rounds_p,
            &self.internal_constants,
            DiffusionMatrixBabyBear
        );

        let mut row = input_state;
        let mut state = input_state.clone();

        // The first half of the external rounds.
        let rounds_f_half = self.rounds_f / 2;
        for r in 0..rounds_f_half {
            poseidon2.add_rc(state, &self.external_constants[r]);
            poseidon2.sbox(state);
            poseidon2.external_linear_layer.permute_mut(state);
            row.extend(state.iter());
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            state[0] += self.internal_constants[r];
            state[0] = poseidon2.sbox_p(&state[0]);
            poseidon2.internal_linear_layer.permute_mut(state);
            row.extend(state.iter());
        }

        // The second half of the external rounds.
        for r in rounds_f_half..self.rounds_f {
            poseidon2.add_rc(state, &self.external_constants[r]);
            poseidon2.sbox(state);
            poseidon2.external_linear_layer.permute_mut(state);
            row.extend(state.iter());
        }

        row
    }
}
