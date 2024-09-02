use std::{borrow::Borrow, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use ax_sdk::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use super::super::{
    check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
    utils::{big_uint_to_limbs, secp256k1_prime},
    OverflowInt,
};
use crate::{
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
};

// Testing AIR:
// Constrain: x^2 + y = p * quotient, that is x^2 + y = 0 (mod p)
#[derive(Clone)]
pub struct TestCarryCols<const N: usize, T> {
    // limbs of x, length N.
    pub x: Vec<T>,
    // limbs of y, length 2N.
    pub y: Vec<T>,
    // 2N
    pub carries: Vec<T>,
    // quotient limbs, length is going to be 1 as x^2 , y and p are all 256 bits.
    pub quotient: Vec<T>,
}

impl<const N: usize, T: Clone> TestCarryCols<N, T> {
    pub fn get_width() -> usize {
        5 * N + 1
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0..N].to_vec();
        let y = slc[N..3 * N].to_vec();
        let carries = slc[3 * N..5 * N].to_vec();
        let quotient = slc[5 * N..5 * N + 1].to_vec();

        Self {
            x,
            y,
            quotient,
            carries,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.carries);
        flattened.extend_from_slice(&self.quotient);

        flattened
    }
}

pub struct TestCarryAir<const N: usize> {
    pub test_carry_sub_air: CheckCarryModToZeroSubAir,
    pub modulus: BigUint,
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
        let TestCarryCols {
            x,
            y,
            carries,
            quotient,
        } = cols;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(y, self.limb_bits);
        let expr = (x_overflow.clone() * x_overflow.clone()) + y_overflow.clone();

        self.test_carry_sub_air.constrain_carry_mod_to_zero(
            builder,
            expr,
            CheckCarryModToZeroCols { carries, quotient },
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

        let quotient = (x.clone() * x.clone() + y.clone()) / self.modulus.clone();
        let q_limb = big_uint_to_limbs(quotient.clone(), self.limb_bits);
        for &q in q_limb.iter() {
            range_check(self.limb_bits, q);
        }
        let quotient_f: Vec<F> = q_limb.iter().map(|&x| F::from_canonical_usize(x)).collect();
        let x_overflow = OverflowInt::<isize>::from_big_uint(x, self.limb_bits, Some(N));
        let y_overflow = OverflowInt::<isize>::from_big_uint(y, self.limb_bits, Some(2 * N));
        let q_overflow = OverflowInt::<isize>::from_big_uint(quotient, self.limb_bits, None);
        assert_eq!(q_overflow.limbs.len(), 1);
        let p_overflow =
            OverflowInt::<isize>::from_big_uint(self.modulus.clone(), self.limb_bits, Some(N * 2));
        let expr =
            x_overflow.clone() * x_overflow.clone() + y_overflow.clone() - p_overflow * q_overflow;
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        let carry_min_abs = self
            .test_carry_sub_air
            .check_carry_to_zero
            .carry_min_value_abs as isize;
        for (i, &carry) in carries.iter().enumerate() {
            range_check(
                self.test_carry_sub_air.check_carry_to_zero.carry_bits,
                (carry + carry_min_abs) as usize,
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
            quotient: quotient_f,
            carries: carries_f,
        }
    }
}

// number of limbs of X.
const N: usize = 13;

fn test_x_square_plus_y_mod(x: BigUint, y: BigUint, prime: BigUint) {
    let limb_bits = 10;
    let num_limbs = N;
    // The equation: x^2 + y = 0 (mod p)
    // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N + 2^10
    // overflow bits: limb_bits * 2 + log2(N) => 24
    let max_overflow_bits = 24;

    let range_bus = 1;
    let range_decomp = 16;
    let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus, 1 << range_decomp));
    let check_carry_sub_air = CheckCarryModToZeroSubAir::new(
        prime.clone(),
        limb_bits,
        range_bus,
        range_decomp,
        max_overflow_bits,
    );
    let test_air = TestCarryAir::<N> {
        test_carry_sub_air: check_carry_sub_air,
        modulus: prime,
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

#[test]
fn test_check_carry_mod_zero() {
    let prime = secp256k1_prime();
    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    let x_square = x.clone() * x.clone();
    let mut next_p = prime.clone();
    while next_p < x_square {
        next_p += prime.clone();
    }
    let y = next_p - x_square;
    test_x_square_plus_y_mod(x, y, prime);
}

#[should_panic]
#[test]
fn test_check_carry_mod_zero_fail() {
    let prime = secp256k1_prime();
    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    let x_square = x.clone() * x.clone();
    let mut next_p = prime.clone();
    while next_p < x_square {
        next_p += prime.clone();
    }
    let y = next_p - x_square + BigUint::from_u32(1).unwrap();
    test_x_square_plus_y_mod(x, y, prime);
}
