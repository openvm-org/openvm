use std::{borrow::Borrow, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use afs_test_utils::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use super::super::{
    check_carry_to_zero::{CheckCarryToZeroCols, CheckCarryToZeroSubAir},
    OverflowInt,
};
use crate::{
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
};

// Testing AIR:
// Constrain: x^2 - y = 0
#[derive(Clone)]
pub struct TestCarryCols<const N: usize, T> {
    // limbs of x, length N.
    pub x: Vec<T>,
    // limbs of y, length 2N.
    pub y: Vec<T>,
    // 2N
    pub carries: Vec<T>,
}

impl<const N: usize, T: Clone> TestCarryCols<N, T> {
    pub fn get_width() -> usize {
        5 * N
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0..N].to_vec();
        let y = slc[N..3 * N].to_vec();
        let carries = slc[3 * N..5 * N].to_vec();

        Self { x, y, carries }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.carries);

        flattened
    }
}

pub struct TestCarryAir<const N: usize> {
    pub test_carry_sub_air: CheckCarryToZeroSubAir,
    pub max_overflow_bits: usize,
    pub decomp: usize,
    pub num_limbs: usize,
    pub limb_bits: usize,
}

impl AirConfig for TestCarryAir<N> {
    type Cols<T> = TestCarryCols<N, T>;
}

impl<F: Field, const N: usize> BaseAir<F> for TestCarryAir<N> {
    fn width(&self) -> usize {
        TestCarryCols::<N, F>::get_width()
    }
}

impl<AB: InteractionBuilder, const N: usize> Air<AB> for TestCarryAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let cols = TestCarryCols::<N, AB::Var>::from_slice(local);
        let TestCarryCols { x, y, carries } = cols;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB>(y, self.limb_bits);
        let expr = (x_overflow.clone() * x_overflow.clone()) - y_overflow.clone();

        self.test_carry_sub_air.constrain_carry_to_zero(
            builder,
            expr,
            CheckCarryToZeroCols { carries },
        );
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for TestCarryAir<N> {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        let range_check = |bits: usize, value: usize| {
            let value = value as u32;
            if bits == self.decomp {
                range_checker.add_count(value);
            } else {
                range_checker.add_count(value);
                range_checker.add_count(value + (1 << self.decomp) - (1 << bits));
            }
        };
        let x_overflow = OverflowInt::<isize>::from_big_uint(x, self.limb_bits);
        let y_overflow = OverflowInt::<isize>::from_big_uint(y, self.limb_bits);
        let expr = x_overflow.clone() * x_overflow.clone() - y_overflow.clone();
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        for (i, &carry) in carries.iter().enumerate() {
            range_check(
                self.test_carry_sub_air.carry_bits,
                (carry + (self.test_carry_sub_air.carry_min_value_abs as isize)) as usize,
            );
            carries_f[i] = F::from_canonical_usize(carry.unsigned_abs())
                * if carry >= 0 { F::one() } else { F::neg_one() };
        }

        TestCarryCols {
            x: x_overflow
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            y: y_overflow
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            carries: carries_f,
        }
    }
}

// number of limbs of X. Y will be X^2 and thus 2N limbs.
const N: usize = 13;

#[test]
fn test_check_carry_to_zero() {
    let limb_bits = 10;
    let num_limbs = N;
    // The equation: x^2 - y
    // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N + 2^10
    // overflow bits: limb_bits * 2 + log2(N) => 24
    let max_overflow_bits = 24;

    let range_bus = 1;
    let range_decomp = 16;

    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    let y = x.clone() * x.clone();
    let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus, 1 << range_decomp));
    let check_carry_sub_air =
        CheckCarryToZeroSubAir::new(limb_bits, range_bus, range_decomp, max_overflow_bits);
    let test_air = TestCarryAir::<N> {
        test_carry_sub_air: check_carry_sub_air,
        max_overflow_bits,
        decomp: range_decomp,
        num_limbs,
        limb_bits,
    };
    let row = test_air
        .generate_trace_row((x, y, range_checker.clone()))
        .flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&test_air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&test_air, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}
