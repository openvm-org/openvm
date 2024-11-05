use axvm::intrinsics::IntModN;

pub struct EcPoint {
    pub x: IntModN,
    pub y: IntModN,
}

pub fn add_ne(p1: &EcPoint, p2: &EcPoint) -> EcPoint {
    #[cfg(not(target_os = "zkvm"))]
    {
        let lambda = (p2.y - p1.y) / (p2.x - p1.x);
        let x3 = lambda * lambda - p1.x - p2.x;
        let y3 = lambda * (p1.x - x3) - p1.y;
        EcPoint { x: x3, y: y3 }
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}

pub fn double(p: &EcPoint) -> EcPoint {
    #[cfg(not(target_os = "zkvm"))]
    {
        let lambda = 3 * p.x * p.x / (2 * p.y);
        let x3 = lambda * lambda - 2 * p.x;
        let y3 = lambda * (p.x - x3) - p.y;
        EcPoint { x: x3, y: y3 }
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}
