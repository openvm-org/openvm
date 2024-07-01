use super::columns::Poseidon2Cols;
use super::Poseidon2Air;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;
use std::borrow::Borrow;

impl<const WIDTH: usize, F: Field> BaseAir<F> for Poseidon2Air<WIDTH, F> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for Poseidon2Air<WIDTH, AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let index_map = Poseidon2Cols::index_map(self);
        let poseidon2_cols = Poseidon2Cols::from_slice(local, &index_map);
        let Poseidon2Cols { io, aux } = poseidon2_cols;

        let half_ext_rounds = self.rounds_f / 2;
        for phase1_index in 0..half_ext_rounds {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase1_index == 0 {
                core::array::from_fn(|i| io.input[i].into())
            } else {
                core::array::from_fn(|i| aux.phase1[phase1_index - 1][i].into())
            };
            state = ext_lin_layer::<AB, WIDTH>(state);
            state = add_ext_consts::<AB, WIDTH>(state, phase1_index, &self.external_constants);
            state = sbox::<AB, WIDTH>(state);
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase1[phase1_index][state_index]);
            }
        }

        for phase2_index in 0..self.rounds_p {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase2_index == 0 {
                ext_lin_layer::<AB, WIDTH>(core::array::from_fn(|i| {
                    aux.phase1[half_ext_rounds - 1][i].into()
                }))
            } else {
                int_lin_layer::<AB, WIDTH>(core::array::from_fn(|i| {
                    aux.phase2[phase2_index - 1][i].into()
                }))
            };
            state[0] += self.internal_constants[phase2_index].into();
            state[0] = sbox_p::<AB>(state[0].clone());
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase2[phase2_index][state_index]);
            }
        }

        for phase3_index in 0..(self.rounds_f - half_ext_rounds) {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase3_index == 0 {
                int_lin_layer::<AB, WIDTH>(core::array::from_fn(|i| {
                    aux.phase2[self.rounds_p - 1][i].into()
                }))
            } else {
                ext_lin_layer::<AB, WIDTH>(core::array::from_fn(|i| {
                    aux.phase3[phase3_index - 1][i].into()
                }))
            };
            state = add_ext_consts::<AB, WIDTH>(
                state,
                phase3_index + half_ext_rounds,
                &self.external_constants,
            );
            state = sbox::<AB, WIDTH>(state);

            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(state_elem.clone(), aux.phase3[phase3_index][state_index]);
            }
        }

        let mut state =
            core::array::from_fn(|i| aux.phase3[self.rounds_f - half_ext_rounds - 1][i].into());
        state = ext_lin_layer::<AB, WIDTH>(state);
        for (state_index, state_elem) in state.iter().enumerate() {
            builder.assert_eq(state_elem.clone(), io.output[state_index]);
        }
    }
}

/// External linear layer. Applies a diffused linear matrix operation, sending every element to every other element
/// Satisfies MDS, also called `Poseidon2ExternalMatrixGeneral`
fn ext_lin_layer<AB: AirBuilder, const WIDTH: usize>(
    state: [AB::Expr; WIDTH],
) -> [AB::Expr; WIDTH] {
    let mut new_state: [<AB as AirBuilder>::Expr; WIDTH] =
        core::array::from_fn(|_| AB::Expr::default());
    for i in (0..WIDTH).step_by(4) {
        let mut sum = AB::Expr::default();
        for j in 0..4 {
            sum += state[i + j].clone();
        }
        for j in 0..4 {
            new_state[i + j] = sum.clone()
                + state[i + j].clone()
                + AB::Expr::two() * state[i + ((j + 1) % 4)].clone();
        }
    }
    let sums: Vec<AB::Expr> = (0..4)
        .map(|j| {
            (0..WIDTH)
                .step_by(4)
                .map(|i| new_state[i + j].clone())
                .sum::<AB::Expr>()
        })
        .collect();
    for i in 0..WIDTH {
        new_state[i] += sums[i % 4].clone();
    }
    new_state
}

/// Internal linear layer. Applies a diffused linear matrix operation, sending every element to every other element
/// Also called `DiffusionMatrixBabyBear`
fn int_lin_layer<AB: AirBuilder, const WIDTH: usize>(
    state: [AB::Expr; WIDTH],
) -> [AB::Expr; WIDTH] {
    let redc_fact = AB::Expr::from_canonical_u32(943718400);
    let sum: AB::Expr = state.clone().into_iter().sum();
    let mut new_state: [<AB as AirBuilder>::Expr; WIDTH] =
        core::array::from_fn(|_| AB::Expr::default());

    new_state[0] = (sum.clone() - state[0].clone() * AB::Expr::two()) * redc_fact.clone();
    for i in 1..WIDTH {
        let fact = if WIDTH == 16 && i == 15 {
            AB::Expr::from_canonical_u32(1 << 15)
        } else {
            AB::Expr::from_canonical_u32(1 << (i - 1))
        };
        new_state[i] = (sum.clone() + state[i].clone() * fact) * redc_fact.clone();
    }
    new_state
}

/// Returns 7th power of field element input
fn sbox_p<AB: AirBuilder>(state_elem: AB::Expr) -> AB::Expr {
    let x2 = state_elem.clone() * state_elem.clone();
    let x4 = x2.clone() * x2.clone();
    let x3 = x2.clone() * state_elem.clone();
    x3 * x4
}

/// Returns elementwise 7th power of vector field element input
fn sbox<AB: AirBuilder, const WIDTH: usize>(state: [AB::Expr; WIDTH]) -> [AB::Expr; WIDTH] {
    core::array::from_fn(|i| sbox_p::<AB>(state[i].clone()))
}

/// Adds external constants elementwise to state, indexed from [[F]]
fn add_ext_consts<AB: AirBuilder, const WIDTH: usize>(
    state: [AB::Expr; WIDTH],
    index: usize,
    external_constants: &[[AB::F; WIDTH]],
) -> [AB::Expr; WIDTH] {
    core::array::from_fn(|i| state[i].clone() + external_constants[index][i])
}
