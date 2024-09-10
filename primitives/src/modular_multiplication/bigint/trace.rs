use std::sync::Arc;

use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    modular_multiplication::{
        bigint::{air::ModularArithmeticBigIntAir, columns::ModularArithmeticBigIntCols},
        columns::ModularMultiplicationCols,
        trace::generate_modular_multiplication_trace_row,
        FullLimbs,
    },
    sub_chip::LocalTraceInstructions,
    var_range::VariableRangeCheckerChip,
};

impl ModularArithmeticBigIntAir {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        pairs: Vec<(BigUint, BigUint)>,
        range_checker: Arc<VariableRangeCheckerChip>,
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

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularArithmeticBigIntAir {
    type LocalInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (a, b, range_checker) = input;
        assert!(a.bits() <= self.total_bits);
        assert!(b.bits() <= self.total_bits);

        let (general, full_limbs) = generate_modular_multiplication_trace_row(
            self.modulus.clone(),
            &self.limb_dimensions,
            range_checker.clone(),
            a,
            b,
        );
        let FullLimbs {
            a_limbs,
            b_limbs,
            r_limbs,
            q_limbs,
        } = full_limbs;
        let [a_limbs, b_limbs, r_limbs] =
            [a_limbs, b_limbs, r_limbs].map(|limbs| limbs.into_iter().flatten().collect_vec());

        let mut sums = vec![0isize; self.num_carries + 1];
        for (i, &a_limb) in a_limbs.iter().enumerate() {
            for (j, &b_limb) in b_limbs.iter().enumerate() {
                sums[i + j] += (a_limb * b_limb) as isize;
            }
        }
        for (i, &p_limb) in self.modulus_limbs.iter().enumerate() {
            for (j, &q_limb) in q_limbs.iter().enumerate() {
                sums[i + j] -= (p_limb * q_limb) as isize;
            }
        }
        for (i, &r_limb) in r_limbs.iter().enumerate() {
            sums[i] -= r_limb as isize;
        }

        let carries = (0..self.num_carries)
            .map(|i| {
                assert_eq!(sums[i] % (1 << self.max_limb_bits), 0);
                let carry = sums[i] >> self.max_limb_bits;
                sums[i + 1] += carry;
                range_checker.add_count(
                    (carry + (self.carry_min_value_abs as isize)) as u32,
                    self.carry_bits,
                );
                F::from_canonical_usize(carry.unsigned_abs())
                    * if carry >= 0 { F::one() } else { F::neg_one() }
            })
            .collect();
        assert_eq!(sums.last(), Some(&0));

        let general = ModularMultiplicationCols::from_slice(
            &general
                .flatten()
                .iter()
                .map(|&x| F::from_canonical_usize(x))
                .collect_vec(),
            &self.limb_dimensions,
        );

        ModularArithmeticBigIntCols { general, carries }
    }
}
