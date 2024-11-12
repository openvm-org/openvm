use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm::intrinsics::{DivUnsafe, IntMod};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::group::Group;

// Secp256k1 modulus
axvm::moduli_setup! {
    IntModN = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
}

axvm::ec_setup! {
    EcPointN = IntModN;
}

impl Group for EcPointN {
    type SelfRef<'a> = &'a Self;

    fn identity() -> Self {
        Self {
            x: IntModN::ZERO,
            y: IntModN::ZERO,
        }
    }

    fn is_identity(&self) -> bool {
        self.x == IntModN::ZERO && self.y == IntModN::ZERO
    }

    fn generator() -> Self {
        unimplemented!()
    }

    fn double(&self) -> Self {
        if self.is_identity() {
            self.clone()
        } else {
            Self::double_impl(self)
        }
    }

    fn double_assign(&mut self) {
        if !self.is_identity() {
            Self::double_assign_impl(self);
        }
    }
}

impl Add<&EcPointN> for EcPointN {
    type Output = Self;

    fn add(self, p2: &EcPointN) -> Self::Output {
        if self.is_identity() {
            p2.clone()
        } else if p2.is_identity() {
            self.clone()
        } else if self.x == p2.x {
            if &self.y + &p2.y == IntModN::ZERO {
                Self::identity()
            } else {
                Self::double_impl(&self)
            }
        } else {
            Self::add_ne(&self, p2)
        }
    }
}

impl Add for EcPointN {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<&EcPointN> for &EcPointN {
    type Output = EcPointN;

    fn add(self, p2: &EcPointN) -> Self::Output {
        if self.is_identity() {
            p2.clone()
        } else if p2.is_identity() {
            self.clone()
        } else if self.x == p2.x {
            if &self.y + &p2.y == IntModN::ZERO {
                EcPointN::identity()
            } else {
                EcPointN::double_impl(self)
            }
        } else {
            EcPointN::add_ne(self, p2)
        }
    }
}

impl AddAssign<&EcPointN> for EcPointN {
    fn add_assign(&mut self, p2: &EcPointN) {
        if self.is_identity() {
            *self = p2.clone();
        } else if p2.is_identity() {
            // do nothing
        } else if self.x == p2.x {
            if &self.y + &p2.y == IntModN::ZERO {
                *self = Self::identity();
            } else {
                Self::double_assign_impl(self);
            }
        } else {
            Self::add_ne_assign(self, p2);
        }
    }
}

impl AddAssign for EcPointN {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl Neg for EcPointN {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: self.x,
            y: -self.y,
        }
    }
}

impl Neg for &EcPointN {
    type Output = EcPointN;

    fn neg(self) -> Self::Output {
        EcPointN {
            x: self.x.clone(),
            y: -self.y.clone(),
        }
    }
}

impl Sub<&EcPointN> for EcPointN {
    type Output = Self;

    fn sub(self, rhs: &EcPointN) -> Self::Output {
        self.add(&rhs.neg())
    }
}

impl Sub for EcPointN {
    type Output = EcPointN;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl Sub<&EcPointN> for &EcPointN {
    type Output = EcPointN;

    fn sub(self, p2: &EcPointN) -> Self::Output {
        self.add(&p2.neg())
    }
}

impl SubAssign<&EcPointN> for EcPointN {
    fn sub_assign(&mut self, p2: &EcPointN) {
        self.add_assign(&p2.neg());
    }
}

impl SubAssign for EcPointN {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}
