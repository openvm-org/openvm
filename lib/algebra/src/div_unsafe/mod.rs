/// Division operation that is undefined behavior when the denominator is not invertible.
pub trait DivUnsafe<Rhs = Self>: Sized {
    /// Output type of `div_unsafe`.
    type Output;

    /// Undefined behavior when denominator is not invertible.
    fn div_unsafe(self, other: Rhs) -> Self::Output;
}

/// Division assignment operation that is undefined behavior when the denominator is not invertible.
pub trait DivAssignUnsafe<Rhs = Self>: Sized {
    /// Undefined behavior when denominator is not invertible.
    fn div_assign_unsafe(&mut self, other: Rhs);
}
