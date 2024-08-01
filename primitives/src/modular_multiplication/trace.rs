use std::collections::VecDeque;
use std::sync::Arc;

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{abs, ToPrimitive};
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::modular_multiplication::air::ModularMultiplicationAir;
use crate::modular_multiplication::columns::{
    ModularMultiplicationAuxCols, ModularMultiplicationCols, ModularMultiplicationIoCols,
    SmallModulusSystemCols,
};
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

impl ModularMultiplicationAir {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        pairs: Vec<(BigUint, BigUint)>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize = BaseAir::<F>::width(self);

        let mut rows = vec![];

        // generate a row for each pair of numbers to multiply
        for (a, b) in pairs {
            let row: Vec<F> = self
                .generate_trace_row((a, b, range_checker.clone()))
                .flatten();
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
        let bit = deque.pop_front().unwrap_or(0);
        bit + (2 * take_limb(deque, limb_size - 1))
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularMultiplicationAir {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (a, b, range_checker) = input;
        assert!(a.bits() <= self.total_bits as u64);
        assert!(b.bits() <= self.total_bits as u64);

        let range_check = |bits: usize, value: usize| {
            let value = value as u32;
            if bits == self.decomp {
                range_checker.add_count(value);
            } else {
                range_checker.add_count(value);
                range_checker.add_count(value << (self.decomp - bits));
            }
        };

        let product = a.clone() * b.clone();
        let r = product.clone() % self.modulus.clone();
        let q = product.clone() / self.modulus.clone();

        let mut a_bits = big_uint_to_bits(a);
        let mut b_bits = big_uint_to_bits(b);
        let mut r_bits = big_uint_to_bits(r);
        let mut q_bits = big_uint_to_bits(q);

        let [(a_elems, a_limbs), (b_elems, b_limbs), (r_elems, r_limbs)] =
            [&mut a_bits, &mut b_bits, &mut r_bits].map(|bits| {
                let elems = self
                    .io_limb_sizes
                    .iter()
                    .map(|limb_sizes_here| {
                        let mut elem = 0;
                        let mut shift = 0;
                        let limbs = limb_sizes_here
                            .iter()
                            .map(|&limb_size| {
                                let limb = take_limb(bits, limb_size);
                                range_check(limb_size, limb);
                                elem += limb << shift;
                                shift += limb_size;
                                limb
                            })
                            .collect();
                        (elem, limbs)
                    })
                    .unzip();
                assert!(bits.is_empty());
                elems
            });

        let q_limbs = self
            .q_limb_sizes
            .iter()
            .map(|&limb_size| {
                let limb = take_limb(&mut q_bits, limb_size);
                range_check(limb_size, limb);
                limb
            })
            .collect();
        assert!(q_bits.is_empty());

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
                    .map(|reduced| {
                        let residue = reduced % small_modulus;
                        let quotient = reduced / small_modulus;
                        range_check(self.small_modulus_bits, residue);
                        range_check(self.quotient_bits, quotient);
                        (residue, quotient)
                    });
                let pq_reduced = system
                    .q_coefficients
                    .iter()
                    .zip_eq(&q_limbs)
                    .map(|(coefficient, limb)| coefficient * limb)
                    .sum::<usize>();
                let total =
                    ((a_residue * b_residue) as isize) - ((pq_reduced + r_reduced) as isize);
                assert_eq!(total % (small_modulus as isize), 0);

                let total_quotient_shifted = (total / (small_modulus as isize))
                    + (1 << self.quotient_bits)
                    - (1 << self.small_modulus_bits);
                range_check(
                    self.quotient_bits,
                    total_quotient_shifted.to_usize().unwrap(),
                );

                let total_quotient_abs = (abs(total) as usize) / small_modulus;
                let total_quotient_abs_elem = F::from_canonical_usize(total_quotient_abs);
                let total_quotient_elem =
                    total_quotient_abs_elem * if total > 0 { F::one() } else { F::neg_one() };
                let total_quotient = total_quotient_elem.as_canonical_u64() as usize;

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
