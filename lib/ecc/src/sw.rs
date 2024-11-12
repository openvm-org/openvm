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

// Secp256k1 curve point
#[derive(Eq, PartialEq, Clone, Debug)]
#[repr(C)]
pub struct EcPointN {
    pub x: IntModN,
    pub y: IntModN,
}

impl EcPointN {
    // Below are wrapper functions for the intrinsic instructions.
    // Should not be called directly.
    #[inline(always)]
    fn add_ne(p1: &EcPointN, p2: &EcPointN) -> EcPointN {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = (&p2.y - &p1.y).div_unsafe(&p2.x - &p1.x);
            let x3 = &lambda * &lambda - &p1.x - &p2.x;
            let y3 = &lambda * &(&p1.x - &x3) - &p1.y;
            EcPointN { x: x3, y: y3 }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<EcPointN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwAddNe as usize,
                uninit.as_mut_ptr(),
                p1 as *const EcPointN,
                p2 as *const EcPointN
            );
            unsafe { uninit.assume_init() }
        }
    }

    #[inline(always)]
    fn add_ne_assign(&mut self, p2: &EcPointN) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = (&p2.y - &self.y).div_unsafe(&p2.x - &self.x);
            let x3 = &lambda * &lambda - &self.x - &p2.x;
            let y3 = &lambda * &(&self.x - &x3) - &self.y;
            self.x = x3;
            self.y = y3;
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwAddNe as usize,
                self as *mut EcPointN,
                self as *const EcPointN,
                p2 as *const EcPointN
            );
        }
    }

    #[inline(always)]
    fn double_impl(p: &EcPointN) -> EcPointN {
        #[cfg(not(target_os = "zkvm"))]
        {
            let two = IntModN::from_u8(2);
            let lambda = &p.x * &p.x * IntModN::from_u8(3).div_unsafe(&p.y * &two);
            let x3 = &lambda * &lambda - &p.x * &two;
            let y3 = &lambda * &(&p.x - &x3) - &p.y;
            EcPointN { x: x3, y: y3 }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<EcPointN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwDouble as usize,
                uninit.as_mut_ptr(),
                p as *const EcPointN,
                "x0"
            );
            unsafe { uninit.assume_init() }
        }
    }

    #[inline(always)]
    fn double_assign_impl(&mut self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let two = IntModN::from_u8(2);
            let lambda = &self.x * &self.x * IntModN::from_u8(3).div_unsafe(&self.y * &two);
            let x3 = &lambda * &lambda - &self.x * &two;
            let y3 = &lambda * &(&self.x - &x3) - &self.y;
            self.x = x3;
            self.y = y3;
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwDouble as usize,
                self as *mut EcPointN,
                self as *const EcPointN,
                "x0"
            );
        }
    }
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
