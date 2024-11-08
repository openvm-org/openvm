#[cfg(target_os = "zkvm")]
use axvm_platform::constants::SwBaseFunct7;

axvm::moduli_setup! {
    IntModN = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
}

#[derive(Eq, PartialEq, Clone)]
#[repr(C)]
pub struct EcPoint {
    pub x: IntModN,
    pub y: IntModN,
}

impl EcPoint {
    pub const IDENTITY: Self = Self {
        x: IntModN::ZERO,
        y: IntModN::ZERO,
    };

    pub fn is_identity(&self) -> bool {
        self.x == Self::IDENTITY.x && self.y == Self::IDENTITY.y
    }

    // Two points can be equal or not.
    pub fn add(p1: &EcPoint, p2: &EcPoint) -> EcPoint {
        if p1.is_identity() {
            p2.clone()
        } else if p2.is_identity() {
            p1.clone()
        } else if p1.x == p2.x {
            if &p1.y + &p2.y == IntModN::ZERO {
                Self::IDENTITY
            } else {
                Self::double(p1)
            }
        } else {
            Self::add_ne(p1, p2)
        }
    }

    #[inline(always)]
    pub fn add_ne(p1: &EcPoint, p2: &EcPoint) -> EcPoint {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = (&p2.y - &p1.y) / (&p2.x - &p1.x);
            let x3 = &lambda * &lambda - &p1.x - &p2.x;
            let y3 = &lambda * &(&p1.x - &x3) - &p1.y;
            EcPoint { x: x3, y: y3 }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<EcPoint> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwAddNe as usize,
                uninit.as_mut_ptr(),
                p1 as *const EcPoint,
                p2 as *const EcPoint
            );
            unsafe { uninit.assume_init() }
        }
    }

    #[inline(always)]
    pub fn add_ne_assign(&mut self, p2: &EcPoint) {
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
                self as *mut EcPoint,
                self as *const EcPoint,
                p2 as *const EcPoint
            );
        }
    }

    #[inline(always)]
    pub fn double(p: &EcPoint) -> EcPoint {
        #[cfg(not(target_os = "zkvm"))]
        {
            let lambda = &p.x * &p.x * 3 / (&p.y * 2);
            let x3 = &lambda * &lambda - &p.x * 2;
            let y3 = &lambda * &(&p.x - &x3) - &p.y;
            EcPoint { x: x3, y: y3 }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<EcPoint> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ShortWeierstrass as usize,
                SwBaseFunct7::SwDouble as usize,
                uninit.as_mut_ptr(),
                p as *const EcPoint,
                "x0"
            );
            unsafe { uninit.assume_init() }
        }
    }

    #[inline(always)]
    pub fn double_assign(&mut self) {
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
                self as *mut EcPoint,
                self as *const EcPoint,
                "x0"
            );
        }
    }
}
