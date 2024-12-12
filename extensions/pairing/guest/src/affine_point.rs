use rand::Rng;

pub trait AffineCoords<F>: Clone {
    /// Creates a new elliptic curve point from its affine coordinates.
    fn new(x: F, y: F) -> Self;

    /// Returns the affine representation x-coordinate of the elliptic curve point.
    fn x(&self) -> F;

    /// Returns the affine representation y-coordinate of the elliptic curve point.
    fn y(&self) -> F;

    /// Negates the elliptic curve point (reflection on the x-axis).
    fn neg(&self) -> Self;

    /// Generates a random elliptic curve point.
    fn random(rng: &mut impl Rng) -> Self;

    /// Returns the generator point of the elliptic curve.
    fn generator() -> Self;

    /// Returns whether the point is the point at infinity or not.
    fn is_infinity(&self) -> bool;
}
