/// A line function on Fp12 x Fp12 in a sparse representation.
/// Let `Fp12 = Fp2[w] / (w^6 - \xi)`. Then the line function is
/// `L(x,y) = 1 + b (x/y) w' + c (1/y) w'^3`
/// where `w' = w` for D-type and `w' = w^{-1}` for M-type twists
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UnevaluatedLine<Fp2> {
    pub b: Fp2,
    pub c: Fp2,
}

/// The output of a line function on Fp12 x Fp12 (see `UnevaluatedLine`).
/// Represents 1 + b w' + c w'^3 where w' = w for D-type and w' = w^-1 for M-type twists.
#[derive(Clone, Copy, Debug)]
pub struct EvaluatedLine<Fp2> {
    pub b: Fp2,
    pub c: Fp2,
}

pub trait Evaluatable<Fp, Fp2> {
    // xy_frac is (x/y, 1/y)
    fn evaluate(&self, xy_frac: &(Fp, Fp)) -> EvaluatedLine<Fp2>;
}

impl<Fp2> IntoIterator for EvaluatedLine<Fp2> {
    type Item = Fp2;
    type IntoIter = core::array::IntoIter<Fp2, 2>;
    fn into_iter(self) -> Self::IntoIter {
        [self.b, self.c].into_iter()
    }
}

/// Convert M-type lines into Fp12 elements
pub trait FromLineMType<Fp2> {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fp2>) -> Self;
}

/// Trait definition for line multiplication opcodes for M-type lines
pub trait LineMulMType<Fp2, Fp12> {
    /// Multiplies two lines in 023-form to get an element in 02345-form
    fn mul_023_by_023(l0: &EvaluatedLine<Fp2>, l1: &EvaluatedLine<Fp2>) -> [Fp2; 5];

    /// Multiplies a line in 023-form with a Fp12 element to get an Fp12 element
    fn mul_by_023(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12;

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_02345(f: &Fp12, x: &[Fp2; 5]) -> Fp12;
}

/// Convert D-type lines into Fp12 elements
pub trait FromLineDType<Fp2> {
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fp2>) -> Self;
}

/// Trait definition for line multiplication opcodes for D-type lines
pub trait LineMulDType<Fp2, Fp12> {
    /// Multiplies two lines in 013-form to get an element in 01234-form
    fn mul_013_by_013(l0: &EvaluatedLine<Fp2>, l1: &EvaluatedLine<Fp2>) -> [Fp2; 5];

    /// Multiplies a line in 013-form with a Fp12 element to get an Fp12 element
    fn mul_by_013(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12;

    /// Multiplies a line in 01234-form with a Fp12 element to get an Fp12 element
    fn mul_by_01234(f: &Fp12, x: &[Fp2; 5]) -> Fp12;
}
