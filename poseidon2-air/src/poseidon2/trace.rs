use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{Poseidon2AuxCols, Poseidon2Cols, Poseidon2IoCols},
    Poseidon2Air,
};
use crate::poseidon2::columns::{Poseidon2ExternalRoundCols, Poseidon2InternalRoundCols};

impl<const WIDTH: usize, F: Field> Poseidon2Air<WIDTH, F> {
    /// Return cached state trace if it exists (input is ignored), otherwise generate trace and return
    ///
    /// TODO: For more efficient trace generation, a custom `DiffusionMatrix` and `ExternalMatrix` should
    /// be provided.
    pub fn generate_trace(&self, input_states: Vec<[F; WIDTH]>) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            input_states
                .into_iter()
                .flat_map(|input_state| {
                    self.generate_local_trace(input_state).flatten().into_iter()
                })
                .collect(),
            self.get_width(),
        )
    }

    /// Cache the trace as a state variable, return the outputs
    pub fn request_trace(&mut self, states: &[[F; WIDTH]]) -> Vec<Vec<F>> {
        states
            .iter()
            .map(|s| self.generate_local_trace(*s).io.output.to_vec())
            .collect()
    }

    /// Perform entire nonlinear external layer operation on state
    pub fn ext_layer(&self, state: &mut [F; WIDTH], constants: &[F; WIDTH]) {
        self.ext_lin_layer(state);
        for (s, c) in state.iter_mut().zip(constants) {
            *s = Self::sbox_p(*s + *c);
        }
    }

    /// Perform entire nonlinear internal layer operation on state
    pub fn int_layer(&self, state: &mut [F; WIDTH], constant: F) {
        self.int_lin_layer(state);
        state[0] += constant;
        state[0] = Self::sbox_p(state[0]);
    }

    /// Generate one row of trace from the input state.
    pub fn generate_local_trace(&self, input_state: [F; WIDTH]) -> Poseidon2Cols<WIDTH, F> {
        let mut state = input_state;

        // The first half of the external rounds.
        let rounds_f_beginning = self.rounds_f / 2;
        let mut phase1 = Vec::with_capacity(rounds_f_beginning);
        for r in 0..rounds_f_beginning {
            self.ext_layer(&mut state, &self.external_constants[r]);
            phase1.push(Poseidon2ExternalRoundCols {
                intermediate_sbox_powers: core::array::from_fn(|_| None),
                round_output: state,
            });
        }

        // The internal rounds.
        let mut phase2 = Vec::with_capacity(self.rounds_p);
        for r in 0..self.rounds_p {
            if r == 0 {
                self.ext_lin_layer(&mut state);
                state[0] += self.internal_constants[0];
                state[0] = Self::sbox_p(state[0]);
            } else {
                self.int_layer(&mut state, self.internal_constants[r]);
            }

            phase2.push(Poseidon2InternalRoundCols {
                intermediate_sbox_power: None,
                round_output: state,
            });
        }

        // The second half of the external rounds.
        let mut phase3 = Vec::with_capacity(self.rounds_f - rounds_f_beginning);
        for r in rounds_f_beginning..self.rounds_f {
            if r == rounds_f_beginning {
                self.int_lin_layer(&mut state);
                for (s, c) in state
                    .iter_mut()
                    .zip(&self.external_constants[rounds_f_beginning])
                {
                    *s = Self::sbox_p(*s + *c);
                }
            } else {
                Self::ext_layer(self, &mut state, &self.external_constants[r]);
            }

            phase3.push(Poseidon2ExternalRoundCols {
                intermediate_sbox_powers: core::array::from_fn(|_| None),
                round_output: state,
            });
        }
        self.ext_lin_layer(&mut state);
        let output_state = state;

        Poseidon2Cols {
            io: Poseidon2IoCols {
                input: input_state,
                output: output_state,
            },
            aux: Poseidon2AuxCols {
                phase1,
                phase2,
                phase3,
            },
        }
    }
}

impl<const WIDTH: usize, F: Field> LocalTraceInstructions<F> for Poseidon2Air<WIDTH, F> {
    type LocalInput = [F; WIDTH];
    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        self.generate_local_trace(local_input)
    }
}
