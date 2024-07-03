pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

use self::columns::Poseidon2Cols;
use p3_field::AbstractField;

/// Air for Poseidon2. Performs a single permutation of the state.
/// Permutation consists of external rounds (linear map combined with nonlinearity),
/// internal rounds, and then the remainder of external rounds.
///
/// This AIR only supports:
/// - sbox of degree 7
/// - WIDTH is multiple of 4 and >= 8
///
/// Spec is at https://hackmd.io/_I1lx-6GROWbKbDi_Vz-pw?view .
pub struct Poseidon2Air<const WIDTH: usize, F: Clone> {
    pub rounds_f: usize,
    pub external_constants: Vec<[F; WIDTH]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<F>,
    /// The M_4 matrix to use for external linear layers.
    pub ext_mds_matrix: [[F; 4]; 4],
    /// The internal linear layers consist of multiplying by matrix of all 1s + diag(int_diag_m1_matrix)
    pub int_diag_m1_matrix: [F; WIDTH],
    pub reduction_factor: F,
    pub bus_index: usize,
}

impl<const WIDTH: usize, F: AbstractField> Poseidon2Air<WIDTH, F> {
    /// MDSMat4 from Plonky3
    /// [ 2 3 1 1 ]
    /// [ 1 2 3 1 ]
    /// [ 1 1 2 3 ]
    /// [ 3 1 1 2 ].
    pub const MDS_MAT_4: [[u32; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
    // Multiply a 4-element vector x by
    // [ 5 7 1 3 ]
    // [ 4 6 1 1 ]
    // [ 1 3 5 7 ]
    // [ 1 1 4 6 ].
    // This uses the formula from the start of Appendix B in the Poseidon2 paper, with multiplications unrolled into additions.
    // It is also the matrix used by the Horizon Labs implementation.
    pub const HL_MDS_MAT_4: [[u32; 4]; 4] =
        [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]];

    pub fn new(
        external_constants: Vec<[F; WIDTH]>,
        internal_constants: Vec<F>,
        ext_mds_matrix: [[u32; 4]; 4],
        int_diag_m1_matrix: [F; WIDTH],
        reduction_factor: F,
        bus_index: usize,
    ) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
            ext_mds_matrix: ext_mds_matrix.map(|row| row.map(F::from_canonical_u32)),
            int_diag_m1_matrix,
            reduction_factor,
            bus_index,
        }
    }

    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, F>::get_width(self)
    }

    // The following are generic in T: AbstractField + From<F> because they are used in the AIR constraints:

    // TODO: allow custom implementations of this via generic DiffusionMatrix: DiffusionPermutation<F, WIDTH> for faster trace generation
    pub fn int_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
        let sum = input.iter().cloned().sum::<T>();
        for (input, diag_m1) in input.iter_mut().zip(&self.int_diag_m1_matrix) {
            *input = (sum.clone() + T::from(diag_m1.clone()) * input.clone())
                * self.reduction_factor.clone().into();
        }
    }

    // TODO: add back custom implementations for faster trace generation
    pub fn ext_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
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

    pub fn sbox_p<T: AbstractField>(value: T) -> T {
        let x2 = value.square();
        let x3 = x2.clone() * value;
        let x4 = x2.clone().square();
        x3 * x4
    }

    /// Returns elementwise 7th power of vector field element input
    fn sbox<T: AbstractField>(state: [T; WIDTH]) -> [T; WIDTH] {
        core::array::from_fn(|i| Self::sbox_p::<T>(state[i].clone()))
    }
}
