use std::borrow::Borrow;

use afs_primitives::sub_chip::{AirConfig, SubAir};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use zkhash::{
    ark_ff::PrimeField as _,
    fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::poseidon2_instance_babybear::{MAT_DIAG16_M_1, RC16},
};

use super::{
    columns::{Poseidon2AuxCols, Poseidon2Cols, Poseidon2IoCols},
    Poseidon2Config,
};

pub const SBOX_DEGREE: usize = 7;

/// Air for Poseidon2. Performs a single permutation of the state.
/// Permutation consists of external rounds (linear map combined with nonlinearity),
/// internal rounds, and then the remainder of external rounds.
///
/// This AIR only supports:
/// - sbox of degree 7
/// - WIDTH is multiple of 4 and >= 8
///
/// Spec is at https://hackmd.io/_I1lx-6GROWbKbDi_Vz-pw?view .
#[derive(Clone, Debug)]
pub struct Poseidon2Air<const WIDTH: usize, F> {
    pub rounds_f: usize,
    pub external_constants: Vec<[F; WIDTH]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<F>,
    /// The M_4 matrix to use for external linear layers.
    pub ext_mds_matrix: [[F; 4]; 4],
    /// The internal linear layers consist of multiplying by matrix of all 1s + diag(int_diag_m1_matrix)
    pub int_diag_m1_matrix: [F; WIDTH],
    pub reduction_factor: F,
    // Maximum constraint degree for the AIR. Must be 3, 5, or 7.
    pub max_constraint_degree: usize,
    pub bus_index: usize,
}

impl<const WIDTH: usize, F: AbstractField> Poseidon2Air<WIDTH, F> {
    pub fn new(
        external_constants: Vec<[F; WIDTH]>,
        internal_constants: Vec<F>,
        ext_mds_matrix: [[u32; 4]; 4],
        int_diag_m1_matrix: [F; WIDTH],
        reduction_factor: F,
        max_constraint_degree: usize,
        bus_index: usize,
    ) -> Self {
        assert!(
            max_constraint_degree == 3 || max_constraint_degree == 5 || max_constraint_degree == 7
        );

        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
            ext_mds_matrix: ext_mds_matrix.map(|row| row.map(F::from_canonical_u32)),
            int_diag_m1_matrix,
            reduction_factor,
            max_constraint_degree,
            bus_index,
        }
    }

    pub fn from_config(
        config: Poseidon2Config<WIDTH, F>,
        max_constraint_degree: usize,
        bus_index: usize,
    ) -> Self {
        Self::new(
            config.external_constants,
            config.internal_constants,
            config.ext_mds_matrix,
            config.int_diag_m1_matrix,
            config.reduction_factor,
            max_constraint_degree,
            bus_index,
        )
    }

    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, F>::width(self)
    }

    // The following are generic in T: AbstractField + From<F> because they are used in the AIR constraints:

    // TODO: allow custom implementations of this via generic DiffusionMatrix: DiffusionPermutation<F, WIDTH> for faster trace generation
    pub(crate) fn int_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
        let sum = input.iter().cloned().sum::<T>();
        for (input, diag_m1) in input.iter_mut().zip(&self.int_diag_m1_matrix) {
            *input = (sum.clone() + T::from(diag_m1.clone()) * input.clone())
                * self.reduction_factor.clone().into();
        }
    }

    // TODO: add back custom implementations for faster trace generation
    pub(crate) fn ext_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
        let mut new_state: [T; WIDTH] = core::array::from_fn(|_| T::zero());
        for i in (0..WIDTH).step_by(4) {
            for index1 in 0..4 {
                for index2 in 0..4 {
                    new_state[i + index1] += T::from(self.ext_mds_matrix[index1][index2].clone())
                        * input[i + index2].clone();
                }
            }
        }

        let sums: [T; 4] = core::array::from_fn(|j| {
            (0..WIDTH)
                .step_by(4)
                .map(|i| new_state[i + j].clone())
                .sum()
        });

        for i in 0..WIDTH {
            new_state[i] += sums[i % 4].clone();
        }

        input.clone_from_slice(&new_state);
    }

    pub(crate) fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
        BabyBear::from_canonical_u64(horizen_babybear.into_bigint().0[0])
    }

    pub(crate) fn horizen_round_consts_16() -> (Vec<[BabyBear; 16]>, Vec<BabyBear>, [BabyBear; 16])
    {
        let p3_rc16: Vec<Vec<BabyBear>> = RC16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| Self::horizen_to_p3(*babybear))
                    .collect()
            })
            .collect();

        let rounds_f = 8;
        let rounds_p = 13;
        let rounds_f_beginning = rounds_f / 2;
        let p_end = rounds_f_beginning + rounds_p;
        let external_round_constants: Vec<[BabyBear; 16]> = p3_rc16[..rounds_f_beginning]
            .iter()
            .chain(p3_rc16[p_end..].iter())
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let internal_round_constants: Vec<BabyBear> = p3_rc16[rounds_f_beginning..p_end]
            .iter()
            .map(|round| round[0])
            .collect();
        let horizen_int_diag: [BabyBear; 16] = {
            let mut array = [BabyBear::zero(); 16];
            for (i, elem) in MAT_DIAG16_M_1.iter().enumerate() {
                array[i] = BabyBear::from_canonical_u32(elem.into_bigint().0[0] as u32);
            }
            array
        };
        (
            external_round_constants,
            internal_round_constants,
            horizen_int_diag,
        )
    }
}

impl Default for Poseidon2Air<16, BabyBear> {
    fn default() -> Self {
        Self::from_config(Poseidon2Config::<16, BabyBear>::default(), 7, 0)
    }
}

impl<const WIDTH: usize, F: Field> BaseAir<F> for Poseidon2Air<WIDTH, F> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<const WIDTH: usize, F> AirConfig for Poseidon2Air<WIDTH, F> {
    type Cols<T> = Poseidon2Cols<WIDTH, T>;
}

impl<AB: InteractionBuilder, const WIDTH: usize> Air<AB> for Poseidon2Air<WIDTH, AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let poseidon2_cols = Poseidon2Cols::from_slice(local, self);
        let Poseidon2Cols { io, aux } = poseidon2_cols;

        SubAir::<AB>::eval(self, builder, io, aux);
    }
}

impl<AB: InteractionBuilder, const WIDTH: usize> SubAir<AB> for Poseidon2Air<WIDTH, AB::F> {
    type IoView = Poseidon2IoCols<WIDTH, AB::Var>;
    type AuxView = Poseidon2AuxCols<WIDTH, AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        self.eval_interactions(builder, io);
        self.eval_without_interactions(builder, io, aux.into_expr::<AB>());
    }
}

impl<const WIDTH: usize, F: Field> Poseidon2Air<WIDTH, F> {
    pub fn eval_without_interactions<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2IoCols<WIDTH, AB::Var>,
        aux: Poseidon2AuxCols<WIDTH, AB::Expr>,
    ) {
        let half_ext_rounds = self.rounds_f / 2;
        for phase1_index in 0..half_ext_rounds {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase1_index == 0 {
                core::array::from_fn(|i| io.input[i].into())
            } else {
                core::array::from_fn(|i| aux.phase1[phase1_index - 1].round_output[i].clone())
            };
            self.ext_lin_layer(&mut state);
            state = add_ext_consts::<AB, WIDTH>(state, phase1_index, &self.external_constants);
            state = self.sbox_air(
                builder,
                state,
                aux.phase1[phase1_index].intermediate_sbox_powers.clone(),
            );
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(
                    state_elem.clone(),
                    aux.phase1[phase1_index].round_output[state_index].clone(),
                );
            }
        }

        for phase2_index in 0..self.rounds_p {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase2_index == 0 {
                let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
                    aux.phase1[half_ext_rounds - 1].round_output[i].clone()
                });
                self.ext_lin_layer(&mut state);
                state
            } else {
                let mut state =
                    core::array::from_fn(|i| aux.phase2[phase2_index - 1].round_output[i].clone());
                self.int_lin_layer(&mut state);
                state
            };
            state[0] += self.internal_constants[phase2_index].into();
            state[0] = self.sbox_p_air(
                builder,
                state[0].clone(),
                aux.phase2[phase2_index].intermediate_sbox_power.clone(),
            );
            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(
                    state_elem.clone(),
                    aux.phase2[phase2_index].round_output[state_index].clone(),
                );
            }
        }

        for phase3_index in 0..(self.rounds_f - half_ext_rounds) {
            // regenerate state as Expr from trace variables on each round
            let mut state = if phase3_index == 0 {
                let mut state =
                    core::array::from_fn(|i| aux.phase2[self.rounds_p - 1].round_output[i].clone());
                self.int_lin_layer(&mut state);
                state
            } else {
                let mut state =
                    core::array::from_fn(|i| aux.phase3[phase3_index - 1].round_output[i].clone());
                self.ext_lin_layer(&mut state);
                state
            };
            state = add_ext_consts::<AB, WIDTH>(
                state,
                phase3_index + half_ext_rounds,
                &self.external_constants,
            );
            state = self.sbox_air(
                builder,
                state,
                aux.phase3[phase3_index].intermediate_sbox_powers.clone(),
            );

            for (state_index, state_elem) in state.iter().enumerate() {
                builder.assert_eq(
                    state_elem.clone(),
                    aux.phase3[phase3_index].round_output[state_index].clone(),
                );
            }
        }

        let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
            aux.phase3[self.rounds_f - half_ext_rounds - 1].round_output[i].clone()
        });
        self.ext_lin_layer(&mut state);
        for (state_index, state_elem) in state.iter().enumerate() {
            builder.assert_eq(state_elem.clone(), io.output[state_index]);
        }
    }

    /// Returns value^SBOX_DEGREE
    fn sbox_p_air<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        value: AB::Expr,
        intermediate_power: Option<AB::Expr>,
    ) -> AB::Expr {
        // When SBOX_DEGREE <= self.max_constraint_degree, we simply compute the SBOX power
        // by repeated multiplication.
        // Otherwise, we make use of the intermediate_power (which is value^self.max_constraint_degree
        // in that case) to reduce the degree used for computing value^SBOX_DEGREE.

        if intermediate_power.is_some() {
            // Ensuring that intermediate_power is value^self.max_constraint_degree
            let mut val_p = AB::Expr::one();
            for _ in 0..self.max_constraint_degree {
                val_p *= value.clone();
            }
            builder.assert_eq(val_p, intermediate_power.clone().unwrap());
        }

        let mut ret = AB::Expr::one();
        for _ in 0..(SBOX_DEGREE - 1) / self.max_constraint_degree {
            ret *= intermediate_power.clone().unwrap();
        }
        for _ in 0..(SBOX_DEGREE - 1) % self.max_constraint_degree + 1 {
            ret *= value.clone();
        }
        ret
    }

    /// Returns elementwise 7th power of vector field element input
    fn sbox_air<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        state: [AB::Expr; WIDTH],
        intermediate_powers: [Option<AB::Expr>; WIDTH],
    ) -> [AB::Expr; WIDTH] {
        state
            .into_iter()
            .zip(intermediate_powers)
            .map(|(state_elem, intermediate_power)| {
                self.sbox_p_air(builder, state_elem, intermediate_power)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
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
