use axvm::intrinsics::IntModN;
#[derive(Eq, PartialEq, Clone)]
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
            todo!()
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
            todo!()
        }
    }
}
