use super::Poseidon2Air;
// use static_assertions::const_assert;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
use p3_symmetric::Permutation;

impl<const WIDTH: usize> Poseidon2Air<WIDTH> {
    // const_assert!(WIDTH == 16 || WIDTH == 24, "WIDTH must be 16 or 24");
    pub fn generate_trace(&self, input_states: Vec<[BabyBear; WIDTH]>) -> RowMajorMatrix<BabyBear>
    where
        DiffusionMatrixBabyBear: Permutation<[BabyBear; WIDTH]>,
    {
        RowMajorMatrix::new(
            input_states
                .iter()
                .flat_map(|input_state| self.generate_local_trace(*input_state))
                .collect(),
            self.get_width(),
        )
    }

    pub fn ext_layer(
        state: &mut [BabyBear; WIDTH],
        constants: &[BabyBear; WIDTH],
        external_layer: &Poseidon2ExternalMatrixGeneral,
    ) where
        DiffusionMatrixBabyBear: Permutation<[BabyBear; WIDTH]>,
    {
        for (s, c) in state.iter_mut().zip(constants) {
            *s = Self::sbox_p(*s + *c);
        }
        external_layer.permute_mut(state);
    }

    pub fn int_layer(
        state: &mut [BabyBear; WIDTH],
        constant: BabyBear,
        internal_layer: &DiffusionMatrixBabyBear,
    ) where
        DiffusionMatrixBabyBear: Permutation<[BabyBear; WIDTH]>,
    {
        state[0] += constant;
        state[0] = Self::sbox_p(state[0]);
        internal_layer.permute_mut(state);
    }

    pub fn sbox_p(value: BabyBear) -> BabyBear {
        let x2 = value.square();
        let x3 = x2 * value;
        let x4 = x2.square();
        x3 * x4
    }

    pub fn generate_local_trace(&self, input_state: [BabyBear; WIDTH]) -> Vec<BabyBear>
    where
        DiffusionMatrixBabyBear: Permutation<[BabyBear; WIDTH]>,
    {
        let mut row = input_state.to_vec();
        let mut state = input_state;

        // The first half of the external rounds.
        let external_layer = Poseidon2ExternalMatrixGeneral {};
        let internal_layer = DiffusionMatrixBabyBear {};
        external_layer.permute_mut(&mut state);
        let rounds_f_half = self.rounds_f / 2;
        for r in 0..rounds_f_half {
            Self::ext_layer(&mut state, &self.external_constants[r], &external_layer);
            row.extend(state.iter());
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            Self::int_layer(&mut state, self.internal_constants[r], &internal_layer);
            row.extend(state.iter());
        }

        // The second half of the external rounds.
        for r in rounds_f_half..self.rounds_f {
            Self::ext_layer(&mut state, &self.external_constants[r], &external_layer);
            row.extend(state.iter());
        }

        row
    }
}
