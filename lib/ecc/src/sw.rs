use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use axvm::intrinsics::IntMod;
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

axvm::moduli_setup! {
    IntModN = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
}

use alloc::{vec, vec::Vec};
use core::ops::Neg;

use micromath::F32Ext;

pub trait Group:
    Clone
    + Debug
    + Eq
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + for<'a> Mul<&'a Self::Scalar, Output = Self>
    + for<'a> MulAssign<&'a Self::Scalar>
{
    type Scalar: IntMod;
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self::Scalar, Output = Self>
    where
        Self: 'a;

    fn identity() -> Self;
    fn is_identity(&self) -> bool;
    fn generator() -> Self;

    fn double(&self) -> Self;
}

#[derive(Eq, PartialEq, Clone)]
#[repr(C)]
pub struct EcPointN {
    pub x: IntModN,
    pub y: IntModN,
}

impl EcPointN {
    pub const IDENTITY: Self = Self {
        x: IntModN::ZERO,
        y: IntModN::ZERO,
    };

    pub fn is_identity(&self) -> bool {
        self.x == Self::IDENTITY.x && self.y == Self::IDENTITY.y
    }

    // Two points can be equal or not.
    pub fn add(&self, p2: &EcPointN) -> EcPointN {
        if self.is_identity() {
            p2.clone()
        } else if p2.is_identity() {
            self.clone()
        } else if self.x == p2.x {
            if &self.y + &p2.y == IntModN::ZERO {
                Self::IDENTITY
            } else {
                Self::double_internal(self)
            }
        } else {
            Self::add_ne(self, p2)
        }
    }

    // Two points can be equal or not.
    pub fn add_assign(&mut self, p2: &EcPointN) {
        if self.is_identity() {
            *self = p2.clone();
        } else if p2.is_identity() {
            // do nothing
        } else if self.x == p2.x {
            if &self.y + &p2.y == IntModN::ZERO {
                *self = Self::IDENTITY;
            } else {
                *self = Self::double_internal(self);
            }
        } else {
            Self::add_ne_assign(self, p2);
        }
    }

    pub fn double(&self) -> Self {
        if self.is_identity() {
            self.clone()
        } else {
            Self::double_internal(self)
        }
    }

    pub fn double_assign(&mut self) {
        if !self.is_identity() {
            Self::double_assign_internal(self);
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: -self.y.clone(),
        }
    }

    // Below are wrapper functions for the intrinsic instructions.
    // Should not be called directly.
    #[inline(always)]
    fn add_ne(p1: &EcPointN, p2: &EcPointN) -> EcPointN {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = (&p2.y - &p1.y) / (&p2.x - &p1.x);
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
            let lambda = (&p2.y - &self.y) / (&p2.x - &self.x);
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
    fn double_internal(p: &EcPointN) -> EcPointN {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = &p.x * &p.x * 3 / (&p.y * 2);
            let x3 = &lambda * &lambda - &p.x * 2;
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
    fn double_assign_internal(&mut self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = &self.x * &self.x * 3 / (&self.y * 2);
            let x3 = &lambda * &lambda - &self.x * 2;
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

// Multi-scalar multiplication
// Reference: https://github.com/privacy-scaling-explorations/halo2curves/blob/8771fe5a5d54fc03e74dbc8915db5dad3ab46a83/src/msm.rs#L335
impl EcPointN {
    pub fn msm(coeffs: &[IntModN], bases: &[EcPointN]) -> Self {
        let coeffs: Vec<_> = coeffs.iter().map(|c| c.as_le_bytes()).collect();
        let mut acc = Self::IDENTITY;

        // c: window size. Will group scalars into c-bit windows
        let c = if bases.len() < 4 {
            1
        } else if bases.len() < 32 {
            3
        } else {
            (bases.len() as f32).ln().ceil() as usize
        };

        let field_byte_size = IntModN::NUM_BYTES;

        // OR all coefficients in order to make a mask to figure out the maximum number of bytes used
        // among all coefficients.
        let mut acc_or = vec![0; field_byte_size];
        for coeff in &coeffs {
            for (acc_limb, limb) in acc_or.iter_mut().zip(coeff.as_ref().iter()) {
                *acc_limb |= *limb;
            }
        }
        let max_byte_size = field_byte_size
            - acc_or
                .iter()
                .rev()
                .position(|v| *v != 0)
                .unwrap_or(field_byte_size);
        if max_byte_size == 0 {
            return Self::IDENTITY;
        }
        let number_of_windows = max_byte_size * 8_usize / c + 1;

        for current_window in (0..number_of_windows).rev() {
            for _ in 0..c {
                acc.double_assign();
            }
            let mut buckets = vec![Self::IDENTITY; 1 << (c - 1)];

            for (coeff, base) in coeffs.iter().zip(bases.iter()) {
                let coeff = Self::get_booth_index(current_window, c, coeff);
                if coeff.is_positive() {
                    buckets[coeff as usize - 1].add_assign(base);
                }
                if coeff.is_negative() {
                    buckets[coeff.unsigned_abs() as usize - 1].add_assign(&base.neg());
                }
            }

            // Summation by parts
            // e.g. 3a + 2b + 1c = a +
            //                    (a) + b +
            //                    ((a) + b) + c
            let mut running_sum = Self::IDENTITY;
            for exp in buckets.into_iter().rev() {
                running_sum = Self::add(&exp, &running_sum);
                acc = Self::add(&acc, &running_sum);
            }
        }
        acc
    }

    fn get_booth_index(window_index: usize, window_size: usize, el: &[u8]) -> i32 {
        // Booth encoding:
        // * step by `window` size
        // * slice by size of `window + 1``
        // * each window overlap by 1 bit * append a zero bit to the least significant end
        // Indexing rule for example window size 3 where we slice by 4 bits:
        // `[0, +1, +1, +2, +2, +3, +3, +4, -4, -3, -3 -2, -2, -1, -1, 0]``
        // So we can reduce the bucket size without preprocessing scalars
        // and remembering them as in classic signed digit encoding

        let skip_bits = (window_index * window_size).saturating_sub(1);
        let skip_bytes = skip_bits / 8;

        // fill into a u32
        let mut v: [u8; 4] = [0; 4];
        for (dst, src) in v.iter_mut().zip(el.iter().skip(skip_bytes)) {
            *dst = *src
        }
        let mut tmp = u32::from_le_bytes(v);

        // pad with one 0 if slicing the least significant window
        if window_index == 0 {
            tmp <<= 1;
        }

        // remove further bits
        tmp >>= skip_bits - (skip_bytes * 8);
        // apply the booth window
        tmp &= (1 << (window_size + 1)) - 1;

        let sign = tmp & (1 << window_size) == 0;

        // div ceil by 2
        tmp = (tmp + 1) >> 1;

        // find the booth action index
        if sign {
            tmp as i32
        } else {
            ((!(tmp - 1) & ((1 << window_size) - 1)) as i32).neg()
        }
    }
}
