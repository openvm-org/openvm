use super::EcPoint;
use crate::bigint::{check_carry_mod_to_zero::CheckCarryModToZeroCols, CanonicalUint, LimbConfig};

// Add two disinct points.
pub struct EcAddCols<T, C: LimbConfig> {
    pub io: EcAddIoCols<T, C>,
    pub aux: EcAddAuxCols<T>,
}

pub struct EcAddIoCols<T, C: LimbConfig> {
    pub p1: EcPoint<T, C>,
    pub p2: EcPoint<T, C>,
    pub p3: EcPoint<T, C>,
}

pub struct EcAddAuxCols<T> {
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
        let aux = EcAddAuxCols::from_slice(&slc[6 * num_limbs..], num_limbs);

        Self { io, aux }
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
}

impl<T: Clone> EcAddAuxCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.lambda);
        flattened.extend_from_slice(&self.lambda_check.quotient);
        flattened.extend_from_slice(&self.lambda_check.carries);
        flattened.extend_from_slice(&self.x3_check.quotient);
        flattened.extend_from_slice(&self.x3_check.carries);
        flattened.extend_from_slice(&self.y3_check.quotient);
        flattened.extend_from_slice(&self.y3_check.carries);

        flattened
    }

    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        let lambda = slc[0..num_limbs].to_vec();
        let lambda_check_q = slc[num_limbs..2 * num_limbs].to_vec();
        let lambda_check_c = slc[2 * num_limbs..4 * num_limbs - 1].to_vec();
        let x3_check_q = slc[4 * num_limbs - 1..5 * num_limbs - 1].to_vec();
        let x3_check_c = slc[5 * num_limbs - 1..7 * num_limbs - 2].to_vec();
        let y3_check_q = slc[7 * num_limbs - 2..8 * num_limbs - 2].to_vec();
        let y3_check_c = slc[8 * num_limbs - 2..10 * num_limbs - 3].to_vec();

        Self {
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
}
