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
    check_carry_to_zero::{
        get_carry_max_abs_and_bits, CheckCarryToZeroCols, CheckCarryToZeroSubAir,
    },
    CanonicalUint, DefaultLimbConfig, OverflowInt,
};
use crate::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir,
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

    pub is_valid: T,
}

impl<const N: usize, T: Clone> TestCarryCols<N, T> {
    pub fn get_width() -> usize {
        5 * N + 1
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0..N].to_vec();
        let y = slc[N..3 * N].to_vec();
        let carries = slc[3 * N..5 * N].to_vec();
        let is_valid = slc[5 * N].clone();
        Self {
            x,
            y,
            carries,
            is_valid,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.carries);
        flattened.push(self.is_valid.clone());
        flattened
    }
}

pub struct TestCarryAir<const N: usize> {
    pub test_carry_sub_air: CheckCarryToZeroSubAir,
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
            is_valid,
        } = cols;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(y, self.limb_bits);
        let expr = (x_overflow.clone() * x_overflow.clone()) - y_overflow.clone();

        self.test_carry_sub_air
            .eval(builder, (expr, CheckCarryToZeroCols { carries }, is_valid));
    }
}

impl TestCarryAir<N> {
    fn generate_trace_row<F: Field>(
        &self,
        (x, y, range_checker): (BigUint, BigUint, &VariableRangeCheckerChip),
    ) -> TestCarryCols<N, F> {
        let x_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&x, Some(N));
        let x_overflow: OverflowInt<isize> = x_canonical.into();
        let y_canonical = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&y, Some(2 * N));
        let y_overflow: OverflowInt<isize> = y_canonical.into();
        assert_eq!(x_overflow.limbs.len(), N);
        assert_eq!(y_overflow.limbs.len(), 2 * N);
        let expr = x_overflow.clone() * x_overflow.clone() - y_overflow.clone();
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        for (i, &carry) in carries.iter().enumerate() {
            range_checker.add_count((carry + (carry_min_abs as isize)) as u32, carry_bits);
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
            is_valid: F::one(),
        }
    }
}

// number of limbs of X, assuming it's 128 bits and limb bits is 10.
const N: usize = 16;

fn test_x_square_minus_y(x: BigUint, y: BigUint) {
    let limb_bits = 8;
    let num_limbs = N;
    // The equation: x^2 - y
    // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N + 2^10
    // overflow bits: limb_bits * 2 + log2(N) => 24
    let range_bus = 1;
    let range_decomp = 16;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let field_element_bits = 30;
    let check_carry_sub_air =
        CheckCarryToZeroSubAir::new(limb_bits, range_bus, range_decomp, field_element_bits);
    let test_air = TestCarryAir::<N> {
        test_carry_sub_air: check_carry_sub_air,
        field_element_bits,
        decomp: range_decomp,
        num_limbs,
        limb_bits,
    };
    let row = test_air
        .generate_trace_row((x, y, &range_checker))
        .flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&test_air));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![test_air, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_check_carry_to_zero() {
    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    let y = x.clone() * x.clone();
    test_x_square_minus_y(x, y);
}

#[should_panic]
#[test]
fn test_check_carry_to_zero_fail() {
    let mut rng = create_seeded_rng();
    let x_len = 4; // in bytes -> 128 bits.
    let x_bytes = (0..x_len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x_bytes);
    let y = x.clone() * x.clone() + BigUint::from_u32(1).unwrap();
    test_x_square_minus_y(x, y);
}
