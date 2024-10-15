use p3_field::AbstractField;

use super::{air::EcAirConfig, EcPoint};
use crate::bigint::{check_carry_mod_to_zero::CheckCarryModToZeroCols, CanonicalUint, LimbConfig};

// Add two disinct points.
#[derive(Clone)]
pub struct EcAddCols<T, C: LimbConfig> {
    pub io: EcAddIoCols<T, C>,
    pub aux: EcAuxCols<T>,
}

// Double a point.
#[derive(Clone)]
pub struct EcDoubleCols<T, C: LimbConfig> {
    pub io: EcDoubleIoCols<T, C>,
    pub aux: EcAuxCols<T>,
}

#[derive(Clone)]
pub struct EcAddIoCols<T, C: LimbConfig> {
    pub p1: EcPoint<T, C>,
    pub p2: EcPoint<T, C>,
    pub p3: EcPoint<T, C>,
}

#[derive(Clone)]
pub struct EcDoubleIoCols<T, C: LimbConfig> {
    pub p1: EcPoint<T, C>,
    pub p2: EcPoint<T, C>,
}

#[derive(Clone)]
pub struct EcAuxCols<T> {
    pub is_valid: T,
    pub lambda: Vec<T>,
    pub lambda_check: CheckCarryModToZeroCols<T>,

    pub x3_check: CheckCarryModToZeroCols<T>,

    pub y3_check: CheckCarryModToZeroCols<T>,
}

impl<T: Clone, C: LimbConfig> EcAddCols<T, C> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.io.flatten());
        flattened.extend_from_slice(&self.aux.flatten());

        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let io = EcAddIoCols::from_slice(&slc[..6 * num_limbs], num_limbs);
        let aux = EcAuxCols::from_slice(&slc[6 * num_limbs..], num_limbs);

        Self { io, aux }
    }

    pub fn width(config: &EcAirConfig) -> usize {
        EcAddIoCols::<T, C>::width(config) + EcAuxCols::<T>::width(config)
    }
}

impl<T: Clone, C: LimbConfig> EcDoubleCols<T, C> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.io.flatten());
        flattened.extend_from_slice(&self.aux.flatten());

        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let io = EcDoubleIoCols::from_slice(&slc[..4 * num_limbs], num_limbs);
        let aux = EcAuxCols::from_slice(&slc[4 * num_limbs..], num_limbs);

        Self { io, aux }
    }

    pub fn width(config: &EcAirConfig) -> usize {
        EcDoubleIoCols::<T, C>::width(config) + EcAuxCols::<T>::width(config)
    }
}

impl<T: Clone, C: LimbConfig> EcAddIoCols<T, C> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.p1.x.limbs);
        flattened.extend_from_slice(&self.p1.y.limbs);
        flattened.extend_from_slice(&self.p2.x.limbs);
        flattened.extend_from_slice(&self.p2.y.limbs);
        flattened.extend_from_slice(&self.p3.x.limbs);
        flattened.extend_from_slice(&self.p3.y.limbs);

        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let x1 = slc[0..num_limbs].to_vec();
        let y1 = slc[num_limbs..2 * num_limbs].to_vec();
        let x2 = slc[2 * num_limbs..3 * num_limbs].to_vec();
        let y2 = slc[3 * num_limbs..4 * num_limbs].to_vec();
        let x3 = slc[4 * num_limbs..5 * num_limbs].to_vec();
        let y3 = slc[5 * num_limbs..6 * num_limbs].to_vec();

        let p1 = EcPoint {
            x: CanonicalUint::<T, C>::from_vec(x1),
            y: CanonicalUint::<T, C>::from_vec(y1),
        };
        let p2 = EcPoint {
            x: CanonicalUint::<T, C>::from_vec(x2),
            y: CanonicalUint::<T, C>::from_vec(y2),
        };
        let p3 = EcPoint {
            x: CanonicalUint::<T, C>::from_vec(x3),
            y: CanonicalUint::<T, C>::from_vec(y3),
        };

        Self { p1, p2, p3 }
    }

    pub fn width(config: &EcAirConfig) -> usize {
        6 * config.num_limbs
    }
}

impl<T: Clone, C: LimbConfig> EcDoubleIoCols<T, C> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.p1.x.limbs);
        flattened.extend_from_slice(&self.p1.y.limbs);
        flattened.extend_from_slice(&self.p2.x.limbs);
        flattened.extend_from_slice(&self.p2.y.limbs);

        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let x1 = slc[0..num_limbs].to_vec();
        let y1 = slc[num_limbs..2 * num_limbs].to_vec();
        let x2 = slc[2 * num_limbs..3 * num_limbs].to_vec();
        let y2 = slc[3 * num_limbs..4 * num_limbs].to_vec();

        let p1 = EcPoint {
            x: CanonicalUint::<T, C>::from_vec(x1),
            y: CanonicalUint::<T, C>::from_vec(y1),
        };
        let p2 = EcPoint {
            x: CanonicalUint::<T, C>::from_vec(x2),
            y: CanonicalUint::<T, C>::from_vec(y2),
        };

        Self { p1, p2 }
    }

    pub fn width(config: &EcAirConfig) -> usize {
        4 * config.num_limbs
    }
}

impl<T: Clone> EcAuxCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.lambda);
        flattened.extend_from_slice(&self.lambda_check.quotient);
        flattened.extend_from_slice(&self.lambda_check.carries);
        flattened.extend_from_slice(&self.x3_check.quotient);
        flattened.extend_from_slice(&self.x3_check.carries);
        flattened.extend_from_slice(&self.y3_check.quotient);
        flattened.extend_from_slice(&self.y3_check.carries);
        flattened.push(self.is_valid.clone());
        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let lambda = slc[0..num_limbs].to_vec();
        let lambda_check_q = slc[num_limbs..2 * num_limbs + 1].to_vec();
        let lambda_check_c = slc[2 * num_limbs + 1..4 * num_limbs + 1].to_vec();
        let x3_check_q = slc[4 * num_limbs + 1..5 * num_limbs + 1].to_vec();
        let x3_check_c = slc[5 * num_limbs + 1..7 * num_limbs].to_vec();
        let y3_check_q = slc[7 * num_limbs..8 * num_limbs].to_vec();
        let y3_check_c = slc[8 * num_limbs..10 * num_limbs - 1].to_vec();
        let is_valid = slc[10 * num_limbs - 1].clone();
        Self {
            is_valid,
            lambda,
            lambda_check: CheckCarryModToZeroCols {
                quotient: lambda_check_q,
                carries: lambda_check_c,
            },
            x3_check: CheckCarryModToZeroCols {
                quotient: x3_check_q,
                carries: x3_check_c,
            },
            y3_check: CheckCarryModToZeroCols {
                quotient: y3_check_q,
                carries: y3_check_c,
            },
        }
    }

    pub fn width(config: &EcAirConfig) -> usize {
        // TODO: this only works for 256bits prime, generalize when we need to support other fields.
        // Current assumptions:
        // q and carries for x and y are num_limbs and 2num_limbs - 1 respectively.
        // Lambda is different: in the ec-double case, λ (2 * y1) - (3 * x1^2) = q * p
        // so q can be larger (in abs) be ~ p * 3 and more than 256 bits.
        // so the q have length num_limbs + 1, and makes carris 2num_limbs.
        // So overall:
        // 1                         // is_valid
        // + num_limbs               // lambda
        // + 2* (3num_limbs - 1)     // x and y check
        // + 3num_limbs + 1          // lambda_check
        10 * config.num_limbs
    }
}

impl<F: AbstractField> EcAuxCols<F> {
    pub fn disabled(num_limbs: usize) -> Self {
        EcAuxCols {
            is_valid: F::zero(),
            lambda: vec![F::zero(); num_limbs],
            lambda_check: CheckCarryModToZeroCols {
                quotient: vec![F::zero(); num_limbs + 1],
                carries: vec![F::zero(); 2 * num_limbs],
            },
            x3_check: CheckCarryModToZeroCols {
                quotient: vec![F::zero(); num_limbs],
                carries: vec![F::zero(); 2 * num_limbs - 1],
            },
            y3_check: CheckCarryModToZeroCols {
                quotient: vec![F::zero(); num_limbs],
                carries: vec![F::zero(); 2 * num_limbs - 1],
            },
        }
    }
}
