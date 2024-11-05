pub mod sw;

use axvm::intrinsics::IntModN;
#[derive(Eq, PartialEq, Clone)]
pub struct EcPoint {
    pub x: IntModN,
    pub y: IntModN,
}

impl EcPoint {
    pub fn is_identity(&self) -> bool {
        self.x == IntModN::zero() && self.y == IntModN::zero()
    }
}
