use std::borrow::Cow;

use afs_compiler::{asm::AsmBuilder, ir::Var, util::execute_program};
use afs_primitives::modular_multiplication::bigint::air::ModularMultiplicationBigIntAir;
use afs_test_utils::utils::create_seeded_rng;
use num_bigint_dig::{algorithms::mod_inverse, BigUint};
use num_traits::{abs, signum, FromPrimitive, One, Zero};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::RngCore;

#[allow(dead_code)]
const WORD_SIZE: usize = 1;

#[test]
fn test_compiler_modular_arithmetic_1() {
    let a = BigUint::from_isize(31).unwrap();
    let b = BigUint::from_isize(115).unwrap();

    let r = BigUint::from_isize(31 * 115).unwrap();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_bigint(a);
    let b_var = builder.eval_bigint(b);
    let r_var = builder.mod_mul(&a_var, &b_var);
    let r_check_var = builder.eval_bigint(r);
    builder.assert_bigint_eq(&r_var, &r_check_var);
    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
fn test_compiler_modular_arithmetic_2() {
    let num_digits = 8;

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    assert!(a < ModularMultiplicationBigIntAir::secp256k1_prime());
    assert!(b < ModularMultiplicationBigIntAir::secp256k1_prime());

    let r = (a.clone() * b.clone()) % ModularMultiplicationBigIntAir::secp256k1_prime();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_bigint(a);
    let b_var = builder.eval_bigint(b);
    let r_var = builder.mod_mul(&a_var, &b_var);
    let r_check_var = builder.eval_bigint(r);
    builder.assert_bigint_eq(&r_var, &r_check_var);
    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
fn test_compiler_modular_arithmetic_conditional() {
    let a = BigUint::from_isize(23).unwrap();
    let b = BigUint::from_isize(41).unwrap();

    let r = BigUint::from_isize(23 * 41).unwrap();
    let s = BigUint::from_isize(1000).unwrap();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_bigint(a);
    let b_var = builder.eval_bigint(b);
    let product_var = builder.mod_mul(&a_var, &b_var);
    let r_var = builder.eval_bigint(r);
    let s_var = builder.eval_bigint(s);

    let should_be_1: Var<F> = builder.uninit();
    let should_be_2: Var<F> = builder.uninit();

    builder.if_bigint_eq(&product_var, &r_var).then_or_else(
        |builder| builder.assign(&should_be_1, F::one()),
        |builder| builder.assign(&should_be_1, F::two()),
    );
    builder.if_bigint_eq(&product_var, &s_var).then_or_else(
        |builder| builder.assign(&should_be_2, F::one()),
        |builder| builder.assign(&should_be_2, F::two()),
    );

    builder.assert_var_eq(should_be_1, F::one());
    builder.assert_var_eq(should_be_2, F::two());

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
#[should_panic]
fn test_compiler_modular_arithmetic_negative() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let one = builder.eval_bigint(BigUint::one());
    let one_times_one = builder.mod_mul(&one, &one);
    let zero = builder.eval_bigint(BigUint::zero());
    builder.assert_bigint_eq(&one_times_one, &zero);
    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

struct Fraction {
    num: isize,
    denom: isize,
}

impl Fraction {
    fn new(num: isize, denom: isize) -> Self {
        Self { num, denom }
    }

    fn to_bigint(&self) -> BigUint {
        let sign = signum(self.num) * signum(self.denom);
        let num = BigUint::from_isize(abs(self.num)).unwrap();
        let denom = BigUint::from_isize(abs(self.denom)).unwrap();
        let mut value = num
            * mod_inverse(
                Cow::Borrowed(&denom),
                Cow::Borrowed(&ModularMultiplicationBigIntAir::secp256k1_prime()),
            )
            .unwrap()
            .to_biguint()
            .unwrap();
        if sign == -1 {
            value = ModularMultiplicationBigIntAir::secp256k1_prime() - value;
        }
        value
    }
}

impl From<isize> for Fraction {
    fn from(value: isize) -> Self {
        Self::new(value, 1)
    }
}

struct Point {
    x: Fraction,
    y: Fraction,
}

impl Point {
    fn new(x: impl Into<Fraction>, y: impl Into<Fraction>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
}

fn test_ec_add(point_1: Point, point_2: Point, point_3: Point) {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let x1_var = builder.eval_bigint(point_1.x.to_bigint());
    let y1_var = builder.eval_bigint(point_1.y.to_bigint());
    let x2_var = builder.eval_bigint(point_2.x.to_bigint());
    let y2_var = builder.eval_bigint(point_2.y.to_bigint());
    let x3_check = builder.eval_bigint(point_3.x.to_bigint());
    let y3_check = builder.eval_bigint(point_3.y.to_bigint());

    let (x3_var, y3_var) = builder.ec_add(&(x1_var, y1_var), &(x2_var, y2_var));

    builder.assert_bigint_eq(&x3_var, &x3_check);
    builder.assert_bigint_eq(&y3_var, &y3_check);

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

// tests for x^3 = y^2 + 7

#[test]
fn test_compiler_ec_double() {
    test_ec_add(Point::new(2, 1), Point::new(2, 1), Point::new(32, -181));
}

#[test]
fn test_compiler_ec_ne_add() {
    test_ec_add(Point::new(2, 1), Point::new(32, 181), Point::new(2, -1));
}

#[test]
fn test_compiler_ec_add_to_zero() {
    test_ec_add(Point::new(2, 1), Point::new(2, -1), Point::new(0, 0));
}

#[test]
fn test_compiler_ec_add_zero_left() {
    test_ec_add(Point::new(0, 0), Point::new(2, 1), Point::new(2, 1))
}

#[test]
fn test_compiler_ec_add_zero_right() {
    test_ec_add(Point::new(2, 1), Point::new(0, 0), Point::new(2, 1))
}

#[test]
fn test_compiler_ec_double_zero() {
    test_ec_add(Point::new(0, 0), Point::new(0, 0), Point::new(0, 0))
}
