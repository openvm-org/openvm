use super::columns::Poseidon2Cols;
use super::Poseidon2Air;
use p3_baby_bear::DiffusionMatrixBabyBear;
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
use p3_symmetric::Permutation;

impl<const WIDTH: usize, T: PrimeField> Poseidon2Air<WIDTH, T> {
    pub fn generate_trace(&self, input_states: Vec<[T; WIDTH]>) -> RowMajorMatrix<T>
    where
        DiffusionMatrixBabyBear: Permutation<[T; WIDTH]>,
    {
        if self.trace.is_some() {
            return self.trace.clone().unwrap();
        }
        RowMajorMatrix::new(
            input_states
                .iter()
                .flat_map(|input_state| self.generate_local_trace(*input_state))
                .collect(),
            self.get_width(),
        )
    }

    pub fn request_trace(&mut self, states: &[[T; WIDTH]]) -> Vec<Vec<T>>
    where
        T: PrimeField,
        DiffusionMatrixBabyBear: Permutation<[T; WIDTH]>,
    {
        let index_map = Poseidon2Cols::<WIDTH, T>::index_map(self);
        let traces: Vec<_> = states
            .iter()
            .map(|s| self.generate_local_trace(*s))
            .collect();
        let outputs: Vec<Vec<T>> = traces
            .iter()
            .map(|t| t[index_map.output.clone()].to_vec())
            .collect();

        self.trace = Some(RowMajorMatrix::new(
            traces.iter().flat_map(|t| t.clone()).collect(),
            self.get_width(),
        ));

        outputs
    }

    pub fn ext_layer(
        state: &mut [T; WIDTH],
        constants: &[T; WIDTH],
        external_layer: &Poseidon2ExternalMatrixGeneral,
    ) where
        DiffusionMatrixBabyBear: Permutation<[T; WIDTH]>,
    {
        external_layer.permute_mut(state);
        for (s, c) in state.iter_mut().zip(constants) {
            *s = Self::sbox_p(*s + *c);
        }
    }

    pub fn int_layer(state: &mut [T; WIDTH], constant: T, internal_layer: &DiffusionMatrixBabyBear)
    where
        DiffusionMatrixBabyBear: Permutation<[T; WIDTH]>,
    {
        internal_layer.permute_mut(state);
        state[0] += constant;
        state[0] = Self::sbox_p(state[0]);
    }

    pub fn sbox_p(value: T) -> T {
        let x2 = value.square();
        let x3 = x2 * value;
        let x4 = x2.square();
        x3 * x4
    }

    pub fn generate_local_trace(&self, input_state: [T; WIDTH]) -> Vec<T>
    where
        DiffusionMatrixBabyBear: Permutation<[T; WIDTH]>,
    {
        let mut row = input_state.to_vec();
        let mut state = input_state;

        // The first half of the external rounds.
        let external_layer = Poseidon2ExternalMatrixGeneral {};
        let internal_layer = DiffusionMatrixBabyBear {};
        let rounds_f_half = self.rounds_f / 2;
        for r in 0..rounds_f_half {
            Self::ext_layer(&mut state, &self.external_constants[r], &external_layer);
            row.extend(state.iter());
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            if r == 0 {
                external_layer.permute_mut(&mut state);
                state[0] += self.internal_constants[0];
                state[0] = Self::sbox_p(state[0]);
            } else {
                Self::int_layer(&mut state, self.internal_constants[r], &internal_layer);
            }
            row.extend(state.iter());
        }

        // The second half of the external rounds.
        for r in rounds_f_half..self.rounds_f {
            if r == rounds_f_half {
                internal_layer.permute_mut(&mut state);
                for (s, c) in state
                    .iter_mut()
                    .zip(&self.external_constants[rounds_f_half])
                {
                    *s = Self::sbox_p(*s + *c);
                }
            } else {
                Self::ext_layer(&mut state, &self.external_constants[r], &external_layer);
            }
            row.extend(state.iter());
        }
        external_layer.permute_mut(&mut state);
        row.extend(state.iter());

        assert_eq!(row.len(), self.get_width());

        row
    }
}
