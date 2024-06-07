use p3_air::AirBuilder;
use p3_field::AbstractField;

// Some helpers
#[inline]
pub fn and<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    a * b
}

#[inline]
pub fn or<AB>(a: AB::Expr, b: AB::Expr) -> AB::Expr
where
    AB: AirBuilder,
{
    a.clone() + b.clone() - a * b
}

#[inline]
pub fn implies<AB>(a: AB::Expr, b: AB::Expr) -> AB::Expr
where
    AB: AirBuilder,
{
    or::<AB>(AB::Expr::one() - a, b)
}
