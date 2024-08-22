use std::{borrow::Borrow, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use afs_test_utils::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use super::{
    check_carry_to_zero::{CheckCarryToZeroCols, CheckCarryToZeroSubAir},
    utils::{big_uint_to_bits, take_limb},
    OverflowInt,
};
use crate::{
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
};

// Testing AIR:
// Two bigint variables, x and y, each with N limbs.
// Constrain: x^2 - y^2 = 0
#[derive(Clone)]
pub struct TestCarryCols<const N: usize, T> {
    // limbs of x and y, length N.
    pub x: Vec<T>,
    pub y: Vec<T>,
    // 2N-1
    pub carries: Vec<T>,
}

impl<const N: usize, T: Clone> TestCarryCols<N, T> {
    pub fn get_width() -> usize {
        4 * N - 1
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0..N].to_vec();
        let y = slc[N..2 * N].to_vec();
        let carries = slc[2 * N..4 * N - 1].to_vec();

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
        // make the expr OverflowInt:
        let mut expr_limbs = vec![AB::Expr::zero(); 2 * N - 1];

        // Expr = x^2 - y^2
        for i in 0..N {
            for j in 0..N {
                expr_limbs[i + j] += x[i] * x[j];
                expr_limbs[i + j] -= y[i] * y[j];
            }
        }
        let overflowed = OverflowInt {
            limbs: expr_limbs,
            max_overflow_bits: self.max_overflow_bits,
        };

        self.test_carry_sub_air.constrain_carry_to_zero(
            builder,
            overflowed,
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
        let mut x_bits = big_uint_to_bits(x);
        let x_limbs: Vec<usize> = (0..self.num_limbs)
            .map(|_| take_limb(&mut x_bits, self.limb_bits))
            .collect();
        let mut y_bits = big_uint_to_bits(y);
        let y_limbs: Vec<usize> = (0..self.num_limbs)
            .map(|_| take_limb(&mut y_bits, self.limb_bits))
            .collect();

        // Longer than carries by 1 for the highest carry.
        let mut sums = vec![0isize; self.num_limbs * 2];
        for i in 0..self.num_limbs {
            for j in 0..self.num_limbs {
                sums[i + j] += (x_limbs[i] * x_limbs[j]) as isize;
                sums[i + j] -= (y_limbs[i] * y_limbs[j]) as isize;
            }
        }
        let mut carries = vec![F::zero(); 2 * self.num_limbs - 1];
        for i in 0..(2 * self.num_limbs - 1) {
            assert_eq!(sums[i] % (1 << self.limb_bits), 0);
            let carry = sums[i] >> self.limb_bits;
            sums[i + 1] += carry;
            range_check(
                self.test_carry_sub_air.carry_bits,
                (carry + (self.test_carry_sub_air.carry_min_value_abs as isize)) as usize,
            );
            carries[i] = F::from_canonical_usize(carry.unsigned_abs())
                * if carry >= 0 { F::one() } else { F::neg_one() };
        }
        // Highest carry should be 0.
        assert_eq!(sums[2 * self.num_limbs - 1], 0);

        TestCarryCols {
            x: x_limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect(),
            y: y_limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect(),
            carries,
        }
    }
}

// number of limbs of X and Y (128bits).
const N: usize = 13;

#[test]
fn test_check_carry_to_zero() {
    let limb_bits = 10;
    let num_limbs = N;
    // Computing overflow bits and carry bits:
    // The equation: x^2 - y^2
    // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N * 2
    // overflow bits: limb_bits * 2 + log2(2N) => 25
    let max_overflow_bits = 25;

    let range_bus = 1;
    let range_decomp = 16;

    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    // y := x so that x^2 - y^2 = 0
    let y = x.clone();
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
