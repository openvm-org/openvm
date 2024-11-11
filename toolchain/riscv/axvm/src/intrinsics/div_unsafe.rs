/// Division operation that is undefined behavior when the denominator is not coprime to the modulus.
pub trait DivUnsafe<Rhs = Self>: Sized {
    /// Output type of `div_unsafe`.
    type Output;

    /// Undefined behavior when denominator is not coprime to N.
    fn div_unsafe(self, other: Rhs) -> Self::Output;
}

/// Division assignment operation that is undefined behavior when the denominator is not coprime to the modulus.
pub trait DivAssignUnsafe<Rhs = Self>: Sized {
    /// Undefined behavior when denominator is not coprime to N.
    fn div_assign_unsafe(&mut self, other: Rhs);
}
