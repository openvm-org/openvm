use axvm::intrinsics::IntModN;

#[derive(Eq, PartialEq)]
pub struct EcPoint {
    pub x: IntModN,
    pub y: IntModN,
}

// Two points can be equal or not.
// pub fn add(p1: &EcPoint, p2: &EcPoint) -> EcPoint {
//     let zero = IntModN::zero();
// }

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
