use p3_air::{AirBuilder, VirtualPairCol};
#[cfg(any(feature = "test-traits", test))]
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};

// TODO: Ideally upstream PrimeField implements From<T>
pub trait FieldFrom<T> {
    fn from_val(value: T) -> Self;
}

#[cfg(any(feature = "test-traits", test))]
impl FieldFrom<u8> for BabyBear {
    fn from_val(value: u8) -> Self {
        BabyBear::from_canonical_u8(value)
    }
}

#[cfg(any(feature = "test-traits", test))]
impl FieldFrom<BabyBear> for BabyBear {
    fn from_val(value: BabyBear) -> Self {
        value
    }
}

pub fn to_vcols<F: Field>(cols: &[usize]) -> Vec<VirtualPairCol<F>> {
    cols.iter()
        .copied()
        .map(VirtualPairCol::single_main)
        .collect()
}

pub fn and<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    a * b
}

/// Assumes that a and b are boolean
pub fn or<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    a.clone() + b.clone() - and::<AB>(a, b)
}

/// Assumes that a and b are boolean
pub fn implies<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    or::<AB>(AB::Expr::one() - a, b)
}
