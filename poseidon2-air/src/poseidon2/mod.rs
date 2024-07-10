pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

use self::columns::Poseidon2Cols;
use lazy_static::lazy_static;
use p3_baby_bear::BabyBear;
use p3_baby_bear::POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY;
use p3_field::{AbstractField, PrimeField32};
use zkhash::ark_ff::PrimeField as _;
use zkhash::fields::babybear::FpBabyBear as HorizenBabyBear;
use zkhash::poseidon2::poseidon2_instance_babybear::{MAT_DIAG16_M_1, RC16};

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

pub struct Poseidon2Config<const WIDTH: usize, F: Clone> {
    pub external_constants: Vec<[F; WIDTH]>,
    pub internal_constants: Vec<F>,
    pub ext_mds_matrix: [[u32; 4]; 4],
    pub int_diag_m1_matrix: [F; WIDTH],
    pub reduction_factor: F,
}

impl<const WIDTH: usize, F: AbstractField> Poseidon2Air<WIDTH, F> {
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

    pub fn from_config(config: Poseidon2Config<WIDTH, F>, bus_index: usize) -> Self {
        Self::new(
            config.external_constants,
            config.internal_constants,
            config.ext_mds_matrix,
            config.int_diag_m1_matrix,
            config.reduction_factor,
            bus_index,
        )
    }

    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, F>::get_width(self)
    }

    // The following are generic in T: AbstractField + From<F> because they are used in the AIR constraints:

    // TODO: allow custom implementations of this via generic DiffusionMatrix: DiffusionPermutation<F, WIDTH> for faster trace generation
    fn int_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
        let sum = input.iter().cloned().sum::<T>();
        for (input, diag_m1) in input.iter_mut().zip(&self.int_diag_m1_matrix) {
            *input = (sum.clone() + T::from(diag_m1.clone()) * input.clone())
                * self.reduction_factor.clone().into();
        }
    }

    // TODO: add back custom implementations for faster trace generation
    fn ext_lin_layer<T: AbstractField + From<F>>(&self, input: &mut [T; WIDTH]) {
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

    fn sbox_p<T: AbstractField>(value: T) -> T {
        let x2 = value.square();
        let x3 = x2.clone() * value;
        let x4 = x2.clone().square();
        x3 * x4
    }

    /// Returns elementwise 7th power of vector field element input
    fn sbox<T: AbstractField>(state: [T; WIDTH]) -> [T; WIDTH] {
        core::array::from_fn(|i| Self::sbox_p::<T>(state[i].clone()))
    }

    fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
        BabyBear::from_canonical_u64(horizen_babybear.into_bigint().0[0])
    }

    fn horizen_round_consts_16() -> (Vec<[BabyBear; 16]>, Vec<BabyBear>, [BabyBear; 16]) {
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
pub const HL_MDS_MAT_4: [[u32; 4]; 4] = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]];

impl<F: PrimeField32> Default for Poseidon2Air<16, F> {
    fn default() -> Self {
        Self::from_config(Poseidon2Config::<16, F>::default(), 0)
    }
}

impl Poseidon2Config<16, BabyBear> {
    pub fn horizen_config() -> Self {
        Self {
            external_constants: HL_BABYBEAR_EXT_CONST_16.to_vec(),
            internal_constants: HL_BABYBEAR_INT_CONST_16.to_vec(),
            ext_mds_matrix: HL_MDS_MAT_4,
            int_diag_m1_matrix: *HL_BABYBEAR_INT_DIAG_16,
            reduction_factor: BabyBear::one(),
        }
    }
}

impl<F: PrimeField32> Poseidon2Config<16, F> {
    /// Using HorizenLab's round constants: https://github.com/HorizenLabs/poseidon2
    pub fn new_hl_baby_bear_16() -> Self {
        let external_round_constants_f: Vec<[F; 16]> = HL_BABYBEAR_EXT_CONST_16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()))
                    .collect::<Vec<F>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let internal_round_constants_f: Vec<F> = HL_BABYBEAR_INT_CONST_16
            .iter()
            .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()))
            .collect();

        let horizen_int_diag_f: [F; 16] = HL_BABYBEAR_INT_DIAG_16
            .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()));

        Self {
            external_constants: external_round_constants_f,
            internal_constants: internal_round_constants_f,
            ext_mds_matrix: HL_MDS_MAT_4,
            int_diag_m1_matrix: horizen_int_diag_f,
            reduction_factor: F::one(),
        }
    }

    pub fn new_p3_baby_bear_16() -> Self {
        let external_round_constants_f: Vec<[F; 16]> = HL_BABYBEAR_EXT_CONST_16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()))
                    .collect::<Vec<F>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let internal_round_constants_f: Vec<F> = HL_BABYBEAR_INT_CONST_16
            .iter()
            .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()))
            .collect();

        let p3_int_diag_f: [F; 16] = POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY
            .map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()));

        Self {
            external_constants: external_round_constants_f,
            internal_constants: internal_round_constants_f,
            ext_mds_matrix: MDS_MAT_4,
            int_diag_m1_matrix: p3_int_diag_f,
            reduction_factor: F::from_wrapped_u64(1u64 << 32).inverse(),
        }
    }
}

impl<F: PrimeField32> Default for Poseidon2Config<16, F> {
    fn default() -> Self {
        Self::new_p3_baby_bear_16()
    }
}

lazy_static! {
    static ref HL_BABYBEAR_EXT_CONST_16: Vec<[BabyBear; 16]> =
        Poseidon2Air::<16, BabyBear>::horizen_round_consts_16().0;
    static ref HL_BABYBEAR_INT_CONST_16: Vec<BabyBear> =
        Poseidon2Air::<16, BabyBear>::horizen_round_consts_16().1;
    static ref HL_BABYBEAR_INT_DIAG_16: [BabyBear; 16] =
        Poseidon2Air::<16, BabyBear>::horizen_round_consts_16().2;
}
