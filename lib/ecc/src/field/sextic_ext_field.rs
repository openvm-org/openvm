use super::{Field, FieldExt};

/// Sextic extension field of `F` with irreducible polynomial `X^6 + \xi`.
/// Elements are represented as `c0 + c1 * w` where `w^6 = \xi`, where \xi depends on the twist of the curve.
///
/// Memory alignment follows alignment of `F`.
/// Memory layout is concatenation of `c0` and `c1`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct SexticExtField<F> {
    pub c: [F; 6],
}

impl<F: Field> SexticExtField<F> {
    pub fn new(c: [F; 6]) -> Self {
        Self { c }
    }
}
