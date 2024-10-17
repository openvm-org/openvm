use std::{cell::RefCell, rc::Rc};

use super::Fp2;
use crate::field_expression::{ExprBuilder, FieldVariable, FieldVariableConfig};

/// Field extension of Fp12 defined with coefficients in Fp2. Fp6-equivalent coefficients are c0: (c0, c2, c4), c1: (c1, c3, c5).
pub struct Fp12<C: FieldVariableConfig> {
    pub c0: Fp2<C>,
    pub c1: Fp2<C>,
    pub c2: Fp2<C>,
    pub c3: Fp2<C>,
    pub c4: Fp2<C>,
    pub c5: Fp2<C>,
}

impl<C: FieldVariableConfig> Fp12<C> {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c0 = Fp2::new(builder.clone());
        let c1 = Fp2::new(builder.clone());
        let c2 = Fp2::new(builder.clone());
        let c3 = Fp2::new(builder.clone());
        let c4 = Fp2::new(builder.clone());
        let c5 = Fp2::new(builder);
        Fp12 {
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        }
    }

    pub fn save(&mut self) {
        self.c0.save();
        self.c1.save();
        self.c2.save();
        self.c3.save();
        self.c4.save();
        self.c5.save();
    }

    pub fn add(&mut self, other: &mut Fp12<C>) -> Fp12<C> {
        Fp12 {
            c0: self.c0.add(&mut other.c0),
            c1: self.c1.add(&mut other.c1),
            c2: self.c2.add(&mut other.c2),
            c3: self.c3.add(&mut other.c3),
            c4: self.c4.add(&mut other.c4),
            c5: self.c5.add(&mut other.c5),
        }
    }

    pub fn sub(&mut self, other: &mut Fp12<C>) -> Fp12<C> {
        Fp12 {
            c0: self.c0.sub(&mut other.c0),
            c1: self.c1.sub(&mut other.c1),
            c2: self.c2.sub(&mut other.c2),
            c3: self.c3.sub(&mut other.c3),
            c4: self.c4.sub(&mut other.c4),
            c5: self.c5.sub(&mut other.c5),
        }
    }

    pub fn mul(&mut self, other: &mut Fp12<C>, xi: &mut Fp2<C>) -> Fp12<C> {
        // c0 = cs0co0 + xi(cs1co5 + cs2co4 + cs3co3 + cs4co2 + cs5co1)
        // c1 = cs0co1 + cs1co0 + xi(cs2co5 + cs3co4 + cs4co3 + cs5co2)
        // c2 = cs0co2 + cs1co1 + cs2co0 + xi(cs3co5 + cs4co4 + cs5co3)
        // c3 = cs0co3 + cs1co2 + cs2co1 + cs3co0 + xi(cs4co5 + cs5co4)
        // c4 = cs0co4 + cs1co3 + cs2co2 + cs3co1 + cs4co0 + xi(cs5co5)
        // c5 = cs0co5 + cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
        let mut c0_xi = xi.mul(
            &mut (self
                .c1
                .mul(&mut other.c5)
                .add(&mut self.c1.mul(&mut other.c5))
                .add(&mut self.c2.mul(&mut other.c4))
                .add(&mut self.c3.mul(&mut other.c3))
                .add(&mut self.c4.mul(&mut other.c1))
                .add(&mut self.c5.mul(&mut other.c0))),
        );
        let c0 = self.c0.mul(&mut other.c0).add(&mut c0_xi);

        let mut c1_xi = xi.mul(
            &mut (self
                .c2
                .mul(&mut other.c5)
                .add(&mut self.c2.mul(&mut other.c5))
                .add(&mut self.c3.mul(&mut other.c4))
                .add(&mut self.c4.mul(&mut other.c3))
                .add(&mut self.c5.mul(&mut other.c2))),
        );
        let c1 = self
            .c0
            .mul(&mut other.c1)
            .add(&mut self.c1.mul(&mut other.c0))
            .add(&mut c1_xi);

        let mut c2_xi = xi.mul(
            &mut (self
                .c3
                .mul(&mut other.c5)
                .add(&mut self.c3.mul(&mut other.c5))
                .add(&mut self.c4.mul(&mut other.c4))
                .add(&mut self.c5.mul(&mut other.c3))),
        );
        let c2 = self
            .c0
            .mul(&mut other.c2)
            .add(&mut self.c1.mul(&mut other.c1))
            .add(&mut self.c2.mul(&mut other.c0))
            .add(&mut c2_xi);

        let mut c3_xi = xi.mul(
            &mut (self
                .c4
                .mul(&mut other.c5)
                .add(&mut self.c5.mul(&mut other.c4))),
        );
        let c3 = self
            .c0
            .mul(&mut other.c3)
            .add(&mut self.c1.mul(&mut other.c2))
            .add(&mut self.c2.mul(&mut other.c1))
            .add(&mut self.c3.mul(&mut other.c0))
            .add(&mut c3_xi);

        let mut c4_xi = xi.mul(&mut (self.c5.mul(&mut other.c5)));
        let c4 = self
            .c0
            .mul(&mut other.c4)
            .add(&mut self.c1.mul(&mut other.c3))
            .add(&mut self.c2.mul(&mut other.c2))
            .add(&mut self.c3.mul(&mut other.c1))
            .add(&mut self.c4.mul(&mut other.c0))
            .add(&mut c4_xi);

        let c5 = self
            .c0
            .mul(&mut other.c5)
            .add(&mut self.c1.mul(&mut other.c4))
            .add(&mut self.c2.mul(&mut other.c3))
            .add(&mut self.c3.mul(&mut other.c2))
            .add(&mut self.c4.mul(&mut other.c1))
            .add(&mut self.c5.mul(&mut other.c0));

        Fp12 {
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        }
    }

    pub fn div(&mut self, _other: &mut Fp12<C>) -> Fp12<C> {
        todo!()
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable<C>) -> Fp12<C> {
        Fp12 {
            c0: self.c0.scalar_mul(fp),
            c1: self.c1.scalar_mul(fp),
            c2: self.c2.scalar_mul(fp),
            c3: self.c3.scalar_mul(fp),
            c4: self.c4.scalar_mul(fp),
            c5: self.c5.scalar_mul(fp),
        }
    }
}

#[cfg(test)]
mod tests {
    use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, G1Affine};

    use super::*;
}
