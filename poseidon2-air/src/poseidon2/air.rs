use super::columns::{Poseidon2AuxCols, Poseidon2Cols, Poseidon2IoCols};
use super::Poseidon2Air;
use afs_chips::sub_chip::{AirConfig, SubAir};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;
use std::borrow::Borrow;

impl<const WIDTH: usize, F: Field> BaseAir<F> for Poseidon2Air<WIDTH, F> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<const WIDTH: usize, F: Clone> AirConfig for Poseidon2Air<WIDTH, F> {
    type Cols<T> = Poseidon2Cols<WIDTH, T>;
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for Poseidon2Air<WIDTH, AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let index_map = Poseidon2Cols::index_map(self);
        let poseidon2_cols = Poseidon2Cols::from_slice(local, &index_map);
        let Poseidon2Cols { io, aux } = poseidon2_cols;

        SubAir::<AB>::eval(self, builder, io, aux);
    }
}

impl<AB: AirBuilder, const WIDTH: usize> SubAir<AB> for Poseidon2Air<WIDTH, AB::F> {
    type IoView = Poseidon2IoCols<WIDTH, AB::Var>;
    type AuxView = Poseidon2AuxCols<WIDTH, AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let half_ext_rounds = self.rounds_f / 2;
        for phase1_index in 0..half_ext_rounds {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase1_index == 0 {
                core::array::from_fn(|i| io.input[i].into())
            } else {
                core::array::from_fn(|i| aux.phase1[phase1_index - 1][i].into())
            };
            self.ext_lin_layer(&mut state);
            state = add_ext_consts::<AB, WIDTH>(state, phase1_index, &self.external_constants);
            state = Self::sbox(state);
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase1[phase1_index][state_index]);
            }
        }

        for phase2_index in 0..self.rounds_p {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase2_index == 0 {
                let mut state = core::array::from_fn(|i| aux.phase1[half_ext_rounds - 1][i].into());
                self.ext_lin_layer(&mut state);
                state
            } else {
                let mut state = core::array::from_fn(|i| aux.phase2[phase2_index - 1][i].into());
                self.int_lin_layer(&mut state);
                state
            };
            state[0] += self.internal_constants[phase2_index].into();
            state[0] = Self::sbox_p(state[0].clone());
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase2[phase2_index][state_index]);
            }
        }

        for phase3_index in 0..(self.rounds_f - half_ext_rounds) {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase3_index == 0 {
                let mut state = core::array::from_fn(|i| aux.phase2[self.rounds_p - 1][i].into());
                self.int_lin_layer(&mut state);
                state
            } else {
                let mut state = core::array::from_fn(|i| aux.phase3[phase3_index - 1][i].into());
                self.ext_lin_layer(&mut state);
                state
            };
            state = add_ext_consts::<AB, WIDTH>(
                state,
                phase3_index + half_ext_rounds,
                &self.external_constants,
            );
            state = Self::sbox(state);

            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase3[phase3_index][state_index]);
            }
        }

        let mut state =
            core::array::from_fn(|i| aux.phase3[self.rounds_f - half_ext_rounds - 1][i].into());
        self.ext_lin_layer(&mut state);
        for (state_index, state_elem) in state.iter().enumerate() {
            builder.assert_eq(state_elem.clone(), io.output[state_index]);
        }
    }
}

/// Adds external constants elementwise to state, indexed from [[F]]
fn add_ext_consts<AB: AirBuilder, const WIDTH: usize>(
    state: [AB::Expr; WIDTH],
    index: usize,
    external_constants: &[[AB::F; WIDTH]],
) -> [AB::Expr; WIDTH] {
    core::array::from_fn(|i| state[i].clone() + external_constants[index][i])
}
