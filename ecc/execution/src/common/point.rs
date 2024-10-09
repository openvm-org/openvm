use halo2curves_axiom::ff::Field;

#[derive(Debug, Clone)]
pub struct EcPoint<F> {
    pub x: F,
    pub y: F,
}

impl<F: Field> EcPoint<F> {
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            y: self.y.neg(),
        }
    }
}
