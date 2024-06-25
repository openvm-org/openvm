use super::columns::Poseidon2Cols;
use super::Poseidon2Air;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_field::AbstractField;
use p3_field::Field;
use p3_field::PrimeField;
use p3_matrix::Matrix;
use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
use p3_symmetric::Permutation;
use std::borrow::Borrow;

impl<const WIDTH: usize, AB: AirBuilder> BaseAir<AB::Expr> for Poseidon2Air<WIDTH, AB::Expr> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for Poseidon2Air<WIDTH, AB::Expr> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let poseidon2_cols = Poseidon2Cols::from_slice(local, self);
        let Poseidon2Cols { io, aux } = poseidon2_cols;

        // let mut state = ext_lin_layer::<AB, WIDTH>(io.input);
        // let half_ext_rounds = self.rounds_f / 2;
        // for phase1_index in 0..half_ext_rounds {
        //     state = add_ext_consts::<AB>(state, phase1_index, &self.external_constants);
        //     state = ext_lin_layer::<AB, WIDTH>(sbox::<AB>(state));
        //     for state_index in 0..WIDTH {
        //         builder.assert_eq(state[state_index], aux.phase1[phase1_index][state_index]);
        //     }
        // }

        // state = aux.phase1.last();
        // for phase2_index in 0..self.rounds_p {
        //     state[0] += self.internal_constants[phase2_index];
        //     state = int_lin_layer::<AB, WIDTH>(state);
        //     for state_index in 0..WIDTH {
        //         builder.assert_eq(state[state_index], aux.phase2[phase2_index][state_index]);
        //     }
        // }

        // for phase3_index in 0..self.rounds_p - half_ext_rounds {
        //     state = add_ext_consts::<AB, WIDTH>(
        //         state,
        //         phase3_index + half_ext_rounds,
        //         &self.external_constants,
        //     );
        //     state = ext_lin_layer::<AB, WIDTH>(sbox::<AB>(state));
        //     for state_index in 0..WIDTH {
        //         builder.assert_eq(state[state_index], aux.phase3[phase3_index][state_index]);
        //     }
        // }
    }
}

fn ext_lin_layer<AB: AirBuilder, const WIDTH: usize>(state: Vec<AB::Expr>) -> Vec<AB::Expr> {
    let mut new_state = vec![AB::Expr::default(); WIDTH];
    for i in (0..WIDTH).step_by(4) {
        let mut sum = AB::Expr::default();
        for j in 0..4 {
            sum = sum + state[i + j].clone();
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

fn int_lin_layer<AB: AirBuilder, const WIDTH: usize>(state: Vec<AB::Expr>) -> Vec<AB::Expr> {
    let sum: AB::Expr = state.iter().map(|x| x.clone()).sum();
    let mut new_state = vec![AB::Expr::default(); WIDTH];
    for i in 0..WIDTH {
        let fact = if i == 0 {
            AB::Expr::zero() - AB::Expr::two()
        } else if WIDTH == 16 && i == 15 {
            AB::Expr::from_canonical_u32(1 << 15)
        } else {
            AB::Expr::from_canonical_u32(1 << (i - 1))
        };
        new_state[i] = sum.clone() + state[i].clone() * fact;
    }
    new_state
}

fn sbox_p<AB: AirBuilder>(state_elem: AB::Expr) -> AB::Expr {
    let x2 = state_elem.clone() * state_elem.clone();
    let x3 = x2.clone() * state_elem;
    let x4 = x2.clone() * x2.clone();
    x3 * x4
}

fn sbox<AB: AirBuilder>(state: Vec<AB::Expr>) -> Vec<AB::Expr> {
    state.iter().map(|x| sbox_p::<AB>(x.clone())).collect()
}

fn add_ext_consts<AB: AirBuilder, const WIDTH: usize>(
    state: Vec<AB::Expr>,
    index: usize,
    external_constants: &Vec<[AB::Expr; WIDTH]>,
) -> Vec<AB::Expr> {
    state
        .iter()
        .zip(external_constants[index].iter())
        .map(|(x, c)| x.clone() + c.clone())
        .collect()
}
