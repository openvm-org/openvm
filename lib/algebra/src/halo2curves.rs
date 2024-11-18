use core::ops::{Add, Mul, Sub};

use halo2curves_axiom::ff;

use crate::{field::Field, DivAssignUnsafe, DivUnsafe};

impl<'a, F: ff::Field> DivUnsafe<&'a F> for F {
    type Output = F;

    fn div_unsafe(self, other: &'a F) -> Self::Output {
        self * other.invert().unwrap()
    }
}

impl<'a, F: ff::Field> DivUnsafe<&'a F> for &'a F {
    type Output = F;

    fn div_unsafe(self, other: &'a F) -> Self::Output {
        *self * other.invert().unwrap()
    }
}

impl<F: ff::Field> DivAssignUnsafe for F {
    fn div_assign_unsafe(&mut self, other: Self) {
        *self *= other.invert().unwrap();
    }
}

impl<'a, F: ff::Field> DivAssignUnsafe<&'a F> for F {
    fn div_assign_unsafe(&mut self, other: &'a F) {
        *self *= other.invert().unwrap();
    }
}

impl<F: ff::Field> Field for F
where
    for<'a> &'a F: Add<&'a F, Output = F> + Sub<&'a F, Output = F> + Mul<&'a F, Output = F>,
{
    const ZERO: Self = <F as ff::Field>::ZERO;
    const ONE: Self = <F as ff::Field>::ONE;

    type SelfRef<'a> = &'a F;

    fn square_assign(&mut self) {
        *self = self.square();
    }
}

mod bn254 {
    use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, Fq6};

    use crate::field::{ComplexConjugate, FieldExtension};

    /// FieldExtension for Fq2 with Fq as base field
    impl FieldExtension<Fq> for Fq2 {
        const D: usize = 2;
        type Coeffs = [Fq; 2];

        fn from_coeffs(coeffs: Self::Coeffs) -> Self {
            Fq2 {
                c0: coeffs[0],
                c1: coeffs[1],
            }
        }

        fn to_coeffs(self) -> Self::Coeffs {
            [self.c0, self.c1]
        }

        fn embed(c0: Fq) -> Self {
            Fq2 { c0, c1: Fq::zero() }
        }

        fn frobenius_map(&self, power: usize) -> Self {
            let mut s = *self;
            Fq2::frobenius_map(&mut s, power);
            s
        }

        fn mul_base(&self, rhs: &Fq) -> Self {
            Fq2 {
                c0: self.c0 * rhs,
                c1: self.c1 * rhs,
            }
        }
    }

    impl ComplexConjugate for Fq2 {
        fn conjugate(self) -> Self {
            let mut s = self;
            Fq2::conjugate(&mut s);
            s
        }

        fn conjugate_assign(&mut self) {
            Fq2::conjugate(self);
        }
    }

    /// FieldExtension for Fq12 with Fq6 as base field since halo2curves does not implement `Field` for Fq6.
    impl FieldExtension<Fq2> for Fq12 {
        const D: usize = 6;
        type Coeffs = [Fq2; 6];

        fn from_coeffs(coeffs: Self::Coeffs) -> Self {
            Fq12 {
                c0: Fq6 {
                    c0: coeffs[0],
                    c1: coeffs[2],
                    c2: coeffs[4],
                },
                c1: Fq6 {
                    c0: coeffs[1],
                    c1: coeffs[3],
                    c2: coeffs[5],
                },
            }
        }

        fn to_coeffs(self) -> Self::Coeffs {
            [
                self.c0.c0, self.c1.c0, self.c0.c1, self.c1.c1, self.c0.c2, self.c1.c2,
            ]
        }

        fn embed(c0: Fq2) -> Self {
            let fq6_pt = Fq6 {
                c0,
                c1: Fq2::zero(),
                c2: Fq2::zero(),
            };
            Fq12 {
                c0: fq6_pt,
                c1: Fq6::zero(),
            }
        }

        fn frobenius_map(&self, power: usize) -> Self {
            let mut s = *self;
            Fq12::frobenius_map(&mut s, power);
            s
        }

        fn mul_base(&self, rhs: &Fq2) -> Self {
            let fq6_pt = Fq6 {
                c0: *rhs,
                c1: Fq2::zero(),
                c2: Fq2::zero(),
            };
            Fq12 {
                c0: self.c0 * fq6_pt,
                c1: self.c1 * fq6_pt,
            }
        }
    }

    /// This is complex conjugation of Fq12 over Fq6
    impl ComplexConjugate for Fq12 {
        fn conjugate(self) -> Self {
            let mut s = self;
            Fq12::conjugate(&mut s);
            s
        }

        fn conjugate_assign(&mut self) {
            Fq12::conjugate(self);
        }
    }
}

mod bls12_381 {
    use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, Fq6};

    use crate::field::{ComplexConjugate, FieldExtension};

    /// FieldExtension for Fq2 with Fq as base field
    impl FieldExtension<Fq> for Fq2 {
        const D: usize = 2;
        type Coeffs = [Fq; 2];

        fn from_coeffs(coeffs: [Fq; 2]) -> Self {
            Fq2 {
                c0: coeffs[0],
                c1: coeffs[1],
            }
        }

        fn to_coeffs(self) -> Self::Coeffs {
            [self.c0, self.c1]
        }

        fn embed(c0: Fq) -> Self {
            Fq2 { c0, c1: Fq::zero() }
        }

        fn frobenius_map(&self, power: usize) -> Self {
            if power % 2 == 0 {
                *self
            } else {
                // note: Fq2::frobenius_map is same as Fq2::conjugate
                Fq2::frobenius_map(self)
            }
        }

        fn mul_base(&self, rhs: &Fq) -> Self {
            Fq2 {
                c0: self.c0 * rhs,
                c1: self.c1 * rhs,
            }
        }
    }

    impl ComplexConjugate for Fq2 {
        fn conjugate(self) -> Self {
            Fq2::conjugate(&self)
        }

        fn conjugate_assign(&mut self) {
            *self = Fq2::conjugate(self);
        }
    }

    /// Note that halo2curves does not implement `Field` for Fq6, so we need to implement the intermediate points manually.
    ///
    /// FieldExtension for Fq12 with Fq2 as base field since halo2curves does not implement `Field` for Fq6.
    impl FieldExtension<Fq2> for Fq12 {
        const D: usize = 6;
        type Coeffs = [Fq2; 6];

        fn from_coeffs(coeffs: [Fq2; 6]) -> Self {
            Fq12 {
                c0: Fq6 {
                    c0: coeffs[0],
                    c1: coeffs[2],
                    c2: coeffs[4],
                },
                c1: Fq6 {
                    c0: coeffs[1],
                    c1: coeffs[3],
                    c2: coeffs[5],
                },
            }
        }

        fn to_coeffs(self) -> Self::Coeffs {
            [
                self.c0.c0, self.c1.c0, self.c0.c1, self.c1.c1, self.c0.c2, self.c1.c2,
            ]
        }

        fn embed(base_elem: Fq2) -> Self {
            let fq6_pt = Fq6 {
                c0: base_elem,
                c1: Fq2::zero(),
                c2: Fq2::zero(),
            };
            Fq12 {
                c0: fq6_pt,
                c1: Fq6::zero(),
            }
        }

        /// Raises this element to p^power, where p is prime characteristic of `Self`.
        fn frobenius_map(&self, power: usize) -> Self {
            let mut x = *self;
            for _ in 0..power % 12 {
                x = Fq12::frobenius_map(&x);
            }
            x
        }

        fn mul_base(&self, rhs: &Fq2) -> Self {
            let fq6_pt = Fq6 {
                c0: *rhs,
                c1: Fq2::zero(),
                c2: Fq2::zero(),
            };
            Fq12 {
                c0: self.c0 * fq6_pt,
                c1: self.c1 * fq6_pt,
            }
        }
    }

    /// This is complex conjugation of Fq12 over Fq6
    impl ComplexConjugate for Fq12 {
        fn conjugate(self) -> Self {
            Fq12::conjugate(&self)
        }

        fn conjugate_assign(&mut self) {
            *self = Fq12::conjugate(self);
        }
    }
}
