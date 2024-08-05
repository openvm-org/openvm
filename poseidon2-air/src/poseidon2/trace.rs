use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::Poseidon2Cols, Poseidon2Air};

impl<const WIDTH: usize, F: Field> Poseidon2Air<WIDTH, F> {
    /// Return cached state trace if it exists (input is ignored), otherwise generate trace and return
    ///
    /// TODO: For more efficient trace generation, a custom `DiffusionMatrix` and `ExternalMatrix` should
    /// be provided.
    pub fn generate_trace(&self, input_states: Vec<[F; WIDTH]>) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            input_states
                .into_iter()
                .flat_map(|input_state| self.generate_local_trace(input_state))
                .collect(),
            self.get_width(),
        )
    }

    /// Cache the trace as a state variable, return the outputs
    pub fn request_trace(&mut self, states: &[[F; WIDTH]]) -> Vec<Vec<F>> {
        let index_map = Poseidon2Cols::<WIDTH, F>::index_map(self);
        let traces: Vec<_> = states
            .iter()
            .map(|s| self.generate_local_trace(*s))
            .collect();
        let outputs: Vec<Vec<F>> = traces
            .iter()
            .map(|t| t[index_map.output.clone()].to_vec())
            .collect();

        outputs
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
    pub fn generate_local_trace(&self, input_state: [F; WIDTH]) -> Vec<F> {
        let mut row = input_state.to_vec();
        let mut state = input_state;

        // The first half of the external rounds.
        let rounds_f_beginning = self.rounds_f / 2;
        for r in 0..rounds_f_beginning {
            self.ext_layer(&mut state, &self.external_constants[r]);
            row.extend(state.iter());
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            if r == 0 {
                self.ext_lin_layer(&mut state);
                state[0] += self.internal_constants[0];
                state[0] = Self::sbox_p(state[0]);
            } else {
                self.int_layer(&mut state, self.internal_constants[r]);
            }
            row.extend(state.iter());
        }

        // The second half of the external rounds.
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
            row.extend(state.iter());
        }
        self.ext_lin_layer(&mut state);
        row.extend(state.iter());

        assert_eq!(row.len(), self.get_width());

        row
    }
}

impl<const WIDTH: usize, F: Field> LocalTraceInstructions<F> for Poseidon2Air<WIDTH, F> {
    type LocalInput = [F; WIDTH];
    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        Poseidon2Cols::from_slice(
            self.generate_local_trace(local_input).as_slice(),
            &Poseidon2Cols::<WIDTH, F>::index_map(self),
        )
    }
}
