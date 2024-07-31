use std::collections::VecDeque;

use itertools::Itertools;
use num_bigint::BigUint;
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::modular_multiplication::air::ModularMultiplicationAir;
use crate::modular_multiplication::columns::{
    ModularMultiplicationAuxCols, ModularMultiplicationCols, ModularMultiplicationIoCols,
    SmallModulusSystemCols,
};
use crate::sub_chip::LocalTraceInstructions;

impl ModularMultiplicationAir {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        pairs: Vec<(BigUint, BigUint)>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize = BaseAir::<F>::width(self);

        let mut rows = vec![];

        // generate a row for each pair of numbers to multiply
        for (a, b) in pairs {
            let row: Vec<F> = self.generate_trace_row((a, b)).flatten();
            rows.extend(row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}

fn big_uint_to_bits(x: BigUint) -> VecDeque<usize> {
    let mut result = VecDeque::new();
    for byte in x.to_bytes_le() {
        for i in 0..8 {
            result.push_back(((byte >> i) as usize) & 1);
        }
    }
    result
}

fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    if limb_size == 0 {
        0
    } else {
        let bit = deque.pop_front().unwrap();
        bit + (2 * take_limb(deque, limb_size - 1))
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularMultiplicationAir {
    type LocalInput = (BigUint, BigUint);

    fn generate_trace_row(&self, input: (BigUint, BigUint)) -> Self::Cols<F> {
        let (a, b) = input;

        let product = a.clone() * b.clone();
        let r = product.clone() % self.modulus.clone();
        let q = product.clone() / self.modulus.clone();

        let mut a_bits = big_uint_to_bits(a);
        let mut b_bits = big_uint_to_bits(b);
        let mut r_bits = big_uint_to_bits(r);
        let mut q_bits = big_uint_to_bits(q);

        let [(a_elems, a_limbs), (b_elems, b_limbs), (r_elems, r_limbs)] =
            [&mut a_bits, &mut b_bits, &mut r_bits].map(|bits| {
                self.io_limb_sizes
                    .iter()
                    .map(|limb_sizes_here| {
                        let mut elem = 0;
                        let mut shift = 0;
                        let limbs = limb_sizes_here
                            .iter()
                            .map(|&limb_size| {
                                let limb = take_limb(bits, limb_size);
                                elem += limb << shift;
                                shift += limb_size;
                                limb
                            })
                            .collect();
                        (elem, limbs)
                    })
                    .unzip()
            });

        let q_limbs = self
            .q_limb_sizes
            .iter()
            .map(|&limb_size| take_limb(&mut q_bits, limb_size))
            .collect();

        let system_cols = self
            .small_moduli_systems
            .iter()
            .map(|system| {
                let small_modulus = system.small_modulus;
                let [a_reduced, b_reduced, r_reduced] =
                    [&a_limbs, &b_limbs, &r_limbs].map(|limbs| {
                        system
                            .io_coefficients
                            .iter()
                            .zip_eq(limbs)
                            .map(|(coefficients_here, limbs_here)| {
                                coefficients_here
                                    .iter()
                                    .zip_eq(limbs_here)
                                    .map(|(coefficient, limb)| coefficient * limb)
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
                    });
                let [(a_residue, a_quotient), (b_residue, b_quotient)] = [a_reduced, b_reduced]
                    .map(|reduced| (reduced % small_modulus, reduced / small_modulus));
                let pq_reduced = system
                    .q_coefficients
                    .iter()
                    .zip_eq(&q_limbs)
                    .map(|(coefficient, limb)| coefficient * limb)
                    .sum::<usize>();
                let total = (a_residue * b_residue) - (pq_reduced + r_reduced);
                assert_eq!(total % small_modulus, 0);
                let total_quotient = total / small_modulus;
                SmallModulusSystemCols {
                    a_residue,
                    a_quotient,
                    b_residue,
                    b_quotient,
                    total_quotient,
                }
            })
            .collect();

        let cols_usize = ModularMultiplicationCols {
            io: ModularMultiplicationIoCols {
                a_elems,
                b_elems,
                r_elems,
            },
            aux: ModularMultiplicationAuxCols {
                a_limbs,
                b_limbs,
                r_limbs,
                q_limbs,
                system_cols,
            },
        };

        ModularMultiplicationCols::from_slice(
            &cols_usize
                .flatten()
                .iter()
                .map(|&x| F::from_canonical_usize(x))
                .collect::<Vec<_>>(),
            self,
        )
    }
}
