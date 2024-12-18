use itertools::izip;
use openvm_stark_backend::{
    p3_field::{AbstractField, Field},
    p3_matrix::dense::RowMajorMatrix,
};

use super::{
    air::SBOX_DEGREE,
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
                .flat_map(|input_state| self.generate_trace_row(input_state).flatten().into_iter())
                .collect(),
            self.get_width(),
        )
    }

    /// Cache the trace as a state variable, return the outputs
    pub fn request_trace(&mut self, states: &[[F; WIDTH]]) -> Vec<Vec<F>> {
        states
            .iter()
            .map(|s| self.generate_trace_row(*s).io.output.to_vec())
            .collect()
    }

    /// Perform entire nonlinear external layer operation on state
    pub fn ext_layer(
        &self,
        state: &mut [F; WIDTH],
        constants: &[F; WIDTH],
        intermediate_powers: &mut [Option<F>; WIDTH],
    ) {
        self.ext_lin_layer(state);
        for ((s, c), ip) in state.iter_mut().zip(constants).zip(intermediate_powers) {
            *s = self.sbox_p_gen(*s + *c, ip);
        }
    }

    /// Perform entire nonlinear internal layer operation on state
    pub fn int_layer(
        &self,
        state: &mut [F; WIDTH],
        constant: F,
        intermediate_power: &mut Option<F>,
    ) {
        self.int_lin_layer(state);
        state[0] += constant;
        state[0] = self.sbox_p_gen(state[0], intermediate_power);
    }

    /// Generate one row of trace from the input state.
    pub fn generate_trace_row(&self, input_state: [F; WIDTH]) -> Poseidon2Cols<WIDTH, F> {
        let mut state = input_state;

        // The first half of the external rounds.
        let rounds_f_beginning = self.rounds_f / 2;
        let mut phase1 = Vec::with_capacity(rounds_f_beginning);
        for r in 0..rounds_f_beginning {
            let mut intermediate_powers = core::array::from_fn(|_| None);
            self.ext_layer(
                &mut state,
                &self.external_constants[r],
                &mut intermediate_powers,
            );
            phase1.push(Poseidon2ExternalRoundCols {
                intermediate_sbox_powers: intermediate_powers,
                round_output: state,
            });
        }

        // The internal rounds.
        let mut phase2 = Vec::with_capacity(self.rounds_p);
        for r in 0..self.rounds_p {
            let mut intermediate_power = None;
            if r == 0 {
                self.ext_lin_layer(&mut state);
                state[0] += self.internal_constants[0];
                state[0] = self.sbox_p_gen(state[0], &mut intermediate_power);
            } else {
                self.int_layer(
                    &mut state,
                    self.internal_constants[r],
                    &mut intermediate_power,
                );
            }

            phase2.push(Poseidon2InternalRoundCols {
                intermediate_sbox_power: intermediate_power,
                round_output: state,
            });
        }

        // The second half of the external rounds.
        let mut phase3 = Vec::with_capacity(self.rounds_f - rounds_f_beginning);
        for r in rounds_f_beginning..self.rounds_f {
            let mut intermediate_powers = core::array::from_fn(|_| None);
            if r == rounds_f_beginning {
                self.int_lin_layer(&mut state);
                for (s, c, ip) in izip!(
                    state.iter_mut(),
                    &self.external_constants[rounds_f_beginning],
                    intermediate_powers.iter_mut(),
                ) {
                    *s = self.sbox_p_gen(*s + *c, ip);
                }
            } else {
                self.ext_layer(
                    &mut state,
                    &self.external_constants[r],
                    &mut intermediate_powers,
                );
            }

            phase3.push(Poseidon2ExternalRoundCols {
                intermediate_sbox_powers: intermediate_powers,
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

    fn sbox_p_gen<T: AbstractField>(&self, value: T, intermediate_power: &mut Option<T>) -> T {
        if self.max_constraint_degree < SBOX_DEGREE {
            // In this case, we compute and set intermediate_power to value^max_constraint_degree
            let mut val_p = T::ONE;
            for _ in 0..self.max_constraint_degree {
                val_p *= value.clone();
            }
            *intermediate_power = Some(val_p);
        }

        let mut ret = T::ONE;
        for _ in 0..(SBOX_DEGREE - 1) / self.max_constraint_degree {
            ret *= intermediate_power.clone().unwrap();
        }
        for _ in 0..(SBOX_DEGREE - 1) % self.max_constraint_degree + 1 {
            ret *= value.clone();
        }
        ret
    }
}
