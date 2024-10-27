use std::{borrow::Borrow, sync::Arc};

use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use super::super::{
    check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
    check_carry_to_zero::get_carry_max_abs_and_bits,
    utils::{big_uint_to_limbs, secp256k1_prime},
    CanonicalUint, DefaultLimbConfig, OverflowInt,
};
use crate::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir,
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

    pub is_valid: T,
}

impl<const N: usize, T: Clone> TestCarryCols<N, T> {
    pub fn get_width() -> usize {
        5 * N + 2
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0..N].to_vec();
        let y = slc[N..3 * N].to_vec();
        let carries = slc[3 * N..5 * N].to_vec();
        let quotient = slc[5 * N..5 * N + 1].to_vec();
        let is_valid = slc[5 * N + 1].clone();
        Self {
            x,
            y,
            quotient,
            carries,
            is_valid,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.carries);
        flattened.extend_from_slice(&self.quotient);
        flattened.push(self.is_valid.clone());
        flattened
    }
}

pub struct TestCarryAir<const N: usize> {
    pub test_carry_sub_air: CheckCarryModToZeroSubAir,
    pub modulus: BigUint,
    pub field_element_bits: usize,
    pub decomp: usize,
    pub num_limbs: usize,
    pub limb_bits: usize,
}

impl<F: Field, const N: usize> BaseAirWithPublicValues<F> for TestCarryAir<N> {}
impl<F: Field, const N: usize> PartitionedBaseAir<F> for TestCarryAir<N> {}
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
            is_valid,
        } = cols;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(y, self.limb_bits);
        let expr = (x_overflow.clone() * x_overflow.clone()) + y_overflow.clone();

        self.test_carry_sub_air.eval(
            builder,
            (
                expr,
                CheckCarryModToZeroCols { carries, quotient },
                is_valid,
            ),
        );
    }
}

impl TestCarryAir<N> {
    fn generate_trace_row<F: Field>(
        &self,
        (x, y, range_checker): (BigUint, BigUint, &VariableRangeCheckerChip),
    ) -> TestCarryCols<N, F> {
        let quotient = (x.clone() * x.clone() + y.clone()) / self.modulus.clone();
        let q_limb = big_uint_to_limbs(&quotient, self.limb_bits);
        for &q in q_limb.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        let quotient_f: Vec<F> = q_limb.iter().map(|&x| F::from_canonical_usize(x)).collect();
        let x_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&x, Some(N));
        let x_overflow: OverflowInt<isize> = x_canonical.into();
        let y_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&y, Some(2 * N));
        let y_overflow: OverflowInt<isize> = y_canonical.into();
        let q_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&quotient, None);
        let q_overflow: OverflowInt<isize> = q_canonical.into();
        assert_eq!(q_overflow.limbs.len(), 1);
        let p_canonical =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&self.modulus, Some(N * 2));
        let p_overflow: OverflowInt<isize> = p_canonical.into();
        let expr =
            x_overflow.clone() * x_overflow.clone() + y_overflow.clone() - p_overflow * q_overflow;
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        for (i, &carry) in carries.iter().enumerate() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
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
            is_valid: F::one(),
        }
    }
}

// number of limbs of X.
const N: usize = 16;

fn test_x_square_plus_y_mod(x: BigUint, y: BigUint, prime: BigUint) {
    let limb_bits = 8;
    let num_limbs = N;
    let field_element_bits = 30;

    let range_bus = 1;
    let range_decomp = 16;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let check_carry_sub_air = CheckCarryModToZeroSubAir::new(
        prime.clone(),
        limb_bits,
        range_bus,
        range_decomp,
        field_element_bits,
    );
    let test_air = TestCarryAir::<N> {
        test_carry_sub_air: check_carry_sub_air,
        modulus: prime,
        field_element_bits,
        decomp: range_decomp,
        num_limbs,
        limb_bits,
    };
    let row = test_air
        .generate_trace_row((x, y, &range_checker))
        .flatten();
    println!("row: {}", row.len());
    println!("width: {}", BaseAir::<BabyBear>::width(&test_air));
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&test_air));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![test_air, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

fn get_random_biguint() -> BigUint {
    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    BigUint::new(x_bytes)
}

#[test]
fn test_check_carry_mod_zero() {
    let prime = secp256k1_prime();
    let x = get_random_biguint();
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
    let x = get_random_biguint();
    let x_square = x.clone() * x.clone();
    let mut next_p = prime.clone();
    while next_p < x_square {
        next_p += prime.clone();
    }
    let y = next_p - x_square + BigUint::from_u32(1).unwrap();
    test_x_square_plus_y_mod(x, y, prime);
}
