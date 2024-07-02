pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

use self::columns::Poseidon2Cols;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

/// Air for Poseidon2. Performs a single permutation of the state.
/// Permutation consists of external rounds (linear map combined with nonlinearity),
/// internal rounds, and then the remainder of external rounds.
///
/// Spec is at https://hackmd.io/_I1lx-6GROWbKbDi_Vz-pw?view .
pub struct Poseidon2Air<const WIDTH: usize, T: Clone> {
    pub rounds_f: usize,
    pub external_constants: Vec<[T; WIDTH]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<T>,
    pub ext_mds_matrix: [[u32; 4]; 4],
    pub int_diag_matrix: [u32; WIDTH],
    pub reduction_factor: u32,
    pub bus_index: usize,
    pub trace: Option<RowMajorMatrix<T>>,
}

impl<const WIDTH: usize, T: Clone> Poseidon2Air<WIDTH, T> {
    pub const MDS_MAT_4: [[u32; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
    pub const DIAG_MAT_16: [u32; 16] = [
        2013265921 - 2,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        32768,
    ];

    pub fn new(
        external_constants: Vec<[T; WIDTH]>,
        internal_constants: Vec<T>,
        ext_mds_matrix: [[u32; 4]; 4],
        int_diag_matrix: [u32; WIDTH],
        reduction_factor: u32,
        bus_index: usize,
    ) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
            ext_mds_matrix,
            int_diag_matrix,
            reduction_factor,
            bus_index,
            trace: None,
        }
    }

    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, T>::get_width(self)
    }

    pub fn int_lin_layer<F: AbstractField>(&self, input: &mut [F; WIDTH]) {
        let sum = input.clone().into_iter().sum::<F>();
        let answer: [F; WIDTH] = core::array::from_fn(|i| {
            (sum.clone() + F::from_canonical_u32(self.int_diag_matrix[i]) * input[i].clone())
                * F::from_canonical_u32(self.reduction_factor)
        });

        input.clone_from_slice(&answer);
    }

    pub fn ext_lin_layer<F: AbstractField>(&self, input: &mut [F; WIDTH]) {
        let mut new_state: [F; WIDTH] = core::array::from_fn(|_| F::zero());
        for i in (0..WIDTH).step_by(4) {
            for index1 in 0..4 {
                for index2 in 0..4 {
                    new_state[i + index1] +=
                        F::from_canonical_u32(self.ext_mds_matrix[index1][index2])
                            * input[i + index2].clone();
                }
            }
        }

        let sums: [F; 4] = core::array::from_fn(|j| {
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

    pub fn sbox_p<F: AbstractField>(value: F) -> F {
        let x2 = value.square();
        let x3 = x2.clone() * value;
        let x4 = x2.clone().square();
        x3 * x4
    }

    /// Returns elementwise 7th power of vector field element input
    fn sbox<F: AbstractField>(state: [F; WIDTH]) -> [F; WIDTH] {
        core::array::from_fn(|i| Self::sbox_p::<F>(state[i].clone()))
    }
}
