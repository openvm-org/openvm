use std::sync::Arc;

use itertools::Itertools;
use num_bigint_dig::BigUint;
use num_traits::ToPrimitive;
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    modular_multiplication::{
        cast_primes::{
            air::ModularMultiplicationPrimesAir,
            columns::{ModularMultiplicationPrimesCols, SmallModulusSystemCols},
        },
        trace::generate_modular_multiplication_trace_row,
        FullLimbs,
    },
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

impl<F: PrimeField64> ModularMultiplicationPrimesAir<F> {
    pub fn generate_trace(
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

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularMultiplicationPrimesAir<F> {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (a, b, range_checker) = input;
        assert!(a.bits() <= self.total_bits);
        assert!(b.bits() <= self.total_bits);

        let range_check = |bits: usize, value: usize| {
            let value = value as u32;
            if bits == self.decomp {
                range_checker.add_count(value);
            } else {
                range_checker.add_count(value);
                range_checker.add_count(value + (1 << self.decomp) - (1 << bits));
            }
        };

        let (general, full_limbs) = generate_modular_multiplication_trace_row(
            self.modulus.clone(),
            &self.limb_dimensions,
            range_checker.clone(),
            self.decomp,
            a,
            b,
        );
        let FullLimbs {
            a_limbs,
            b_limbs,
            r_limbs,
            q_limbs,
        } = full_limbs;

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

                SmallModulusSystemCols {
                    a_quotient,
                    b_quotient,
                }
            })
            .collect();

        let cols_usize = ModularMultiplicationPrimesCols {
            general,
            system_cols,
        };

        ModularMultiplicationPrimesCols::from_slice(
            &cols_usize
                .flatten()
                .iter()
                .map(|&x| F::from_canonical_usize(x))
                .collect::<Vec<_>>(),
            self,
        )
    }
}
