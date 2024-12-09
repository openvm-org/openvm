use alloc::{vec, vec::Vec};
use core::ops::{Add, Neg};

use axvm_algebra_guest::IntMod;

use super::Group;

/// Multi-scalar multiplication via Pippenger's algorithm
// Reference: https://github.com/privacy-scaling-explorations/halo2curves/blob/8771fe5a5d54fc03e74dbc8915db5dad3ab46a83/src/msm.rs#L335
// FIXME[jpw]: there are many memcpy in this function
pub fn msm<EcPoint: Group, Scalar: IntMod>(coeffs: &[Scalar], bases: &[EcPoint]) -> EcPoint
where
    for<'a> &'a EcPoint: Add<&'a EcPoint, Output = EcPoint>,
{
    let coeffs: Vec<_> = coeffs.iter().map(|c| c.as_le_bytes()).collect();
    let mut acc = EcPoint::IDENTITY;

    // c: window size. Will group scalars into c-bit windows
    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        // TODO: finetune this if needed
        bases.len().ilog2() as usize
    };

    let field_byte_size = Scalar::NUM_LIMBS;

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
        return EcPoint::IDENTITY;
    }
    let number_of_windows = max_byte_size * 8_usize / c + 1;

    for current_window in (0..number_of_windows).rev() {
        for _ in 0..c {
            acc.double_assign();
        }
        #[derive(Clone)]
        enum Bucket<EcPoint: Group> {
            None,
            Affine(EcPoint),
        }

        impl<EcPoint: Group> Bucket<EcPoint>
        where
            for<'a> &'a EcPoint: Add<&'a EcPoint, Output = EcPoint>,
        {
            fn add_assign(&mut self, other: &EcPoint) {
                match self {
                    Bucket::None => {
                        *self = Bucket::Affine(other.clone());
                    }
                    Bucket::Affine(a) => {
                        a.add_assign(other);
                    }
                }
            }

            fn sub_assign(&mut self, other: &EcPoint) {
                match self {
                    Bucket::None => {
                        *self = Bucket::Affine(other.clone().neg());
                    }
                    Bucket::Affine(a) => {
                        a.sub_assign(other);
                    }
                }
            }

            fn add(self, mut other: EcPoint) -> EcPoint {
                match self {
                    Bucket::None => other.clone(),
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                }
            }
        }

        let mut buckets: Vec<Bucket<EcPoint>> = vec![Bucket::None; 1 << (c - 1)];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_booth_index(current_window, c, coeff);
            if coeff.is_positive() {
                buckets[coeff as usize - 1].add_assign(base);
            }
            if coeff.is_negative() {
                buckets[coeff.unsigned_abs() as usize - 1].sub_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = EcPoint::IDENTITY;
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            acc = acc.add(&running_sum);
        }
    }
    acc
}

// TODO: benchmark to see if this is faster.
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
