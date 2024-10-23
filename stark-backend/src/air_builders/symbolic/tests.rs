use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};

use crate::air_builders::symbolic::symbolic_expression::SymbolicExpression;

type F = BabyBear;
#[test]
fn test_add() {
    let a = SymbolicExpression::IsFirstRow;
    let b = a + F::one();
    assert_add(&b);
    assert_left_operand_const(&b, 1);
    let c_one: SymbolicExpression<F> = F::one().into();
    let c = c_one + b;
    assert_add(&c);
    assert_left_operand_const(&c, 2);
    let d = c + F::one();
    assert_add(&d);
    assert_left_operand_const(&d, 3);
    let const_3: SymbolicExpression<F> = F::from_canonical_u32(3).into();
    let e = d - const_3;
    assert_is_first_row(&e);
}

#[test]
fn test_sub() {
    {
        let a = SymbolicExpression::IsFirstRow;
        let b = a - F::one();
        assert_sub(&b);
        assert_right_operand_const(&b, 1);
        let c = b - F::one();
        assert_sub(&c);
        assert_right_operand_const(&c, 2);
        let const_2: SymbolicExpression<F> = F::two().into();
        let d = const_2 - c;
        assert_sub(&d);
        assert_left_operand_const(&d, 4);
        let const_4: SymbolicExpression<F> = F::from_canonical_u32(4).into();
        let e = const_4 - d;
        assert_is_first_row(&e);
    }
    {
        let a = SymbolicExpression::IsFirstRow;
        let const_1: SymbolicExpression<F> = F::one().into();
        let b = const_1 - a;
        assert_sub(&b);
        assert_left_operand_const(&b, 1);
        let const_2: SymbolicExpression<F> = F::two().into();
        let c = const_2 - b;
        assert_add(&c);
        assert_left_operand_const(&c, 1);
    }
}

#[test]
fn test_mul() {
    let a = SymbolicExpression::IsFirstRow;
    let b = a - F::one();
    assert_sub(&b);
    assert_right_operand_const(&b, 1);
    let const_1: SymbolicExpression<F> = F::one().into();
    let c = b * const_1;
    assert_sub(&c);
    assert_right_operand_const(&c, 1);
    let const_2: SymbolicExpression<F> = F::two().into();
    let d = c * const_2;
    assert_mul(&d);
    assert_left_operand_const(&d, 2);
    let const_0: SymbolicExpression<F> = F::zero().into();
    let e = d * const_0;
    assert_constant(&e, 0);
}

#[test]
fn test_neg() {
    {
        let a = SymbolicExpression::IsFirstRow;
        let b = a - F::one();
        assert_sub(&b);
        assert_right_operand_const(&b, 1);
        let c = -b;
        assert_sub(&c);
        assert_left_operand_const(&c, 1);
    }
    {
        let a = SymbolicExpression::IsFirstRow;
        let b = a + F::one();
        assert_add(&b);
        assert_left_operand_const(&b, 1);
        let c = -b;
        assert_sub(&c);
        assert_left_operand_const_f(&c, -F::one());
    }
}

fn assert_left_operand_const_f<F: Field>(expr: &SymbolicExpression<F>, expected: F) {
    match expr {
        SymbolicExpression::Add { x, .. }
        | SymbolicExpression::Sub { x, .. }
        | SymbolicExpression::Mul { x, .. } => {
            if let SymbolicExpression::Constant(cv) = x.as_ref() {
                assert_eq!(*cv, expected);
            } else {
                panic!("Left operand is not a constant");
            }
        }
        _ => panic!("Unexpected expression"),
    }
}
fn assert_left_operand_const<F: Field>(expr: &SymbolicExpression<F>, expected: u32) {
    assert_left_operand_const_f(expr, F::from_canonical_u32(expected));
}

fn assert_right_operand_const<F: Field>(expr: &SymbolicExpression<F>, expected: u32) {
    match expr {
        SymbolicExpression::Add { y, .. }
        | SymbolicExpression::Sub { y, .. }
        | SymbolicExpression::Mul { y, .. } => {
            if let SymbolicExpression::Constant(cv) = y.as_ref() {
                assert_eq!(*cv, F::from_canonical_u32(expected));
            } else {
                panic!("Right operand is not a constant");
            }
        }
        _ => panic!("Unexpected expression"),
    }
}

fn assert_add<F: Field>(expr: &SymbolicExpression<F>) {
    match &expr {
        SymbolicExpression::Add { .. } => {}
        _ => panic!("Expect Add"),
    }
}

fn assert_sub<F: Field>(expr: &SymbolicExpression<F>) {
    match &expr {
        SymbolicExpression::Sub { .. } => {}
        _ => panic!("Expect Sub"),
    }
}

fn assert_mul<F: Field>(expr: &SymbolicExpression<F>) {
    match &expr {
        SymbolicExpression::Mul { .. } => {}
        _ => panic!("Expect Mul"),
    }
}

fn assert_is_first_row<F: Field>(expr: &SymbolicExpression<F>) {
    match &expr {
        SymbolicExpression::IsFirstRow => {}
        _ => panic!("Expect IsFirstRow"),
    }
}
fn assert_constant<F: Field>(expr: &SymbolicExpression<F>, val: u32) {
    match &expr {
        SymbolicExpression::Constant(cv) => {
            assert_eq!(*cv, F::from_canonical_u32(val));
        }
        _ => panic!("Expect IsFirstRow"),
    }
}
