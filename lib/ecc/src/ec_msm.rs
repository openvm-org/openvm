use std::ops::Neg;

use elliptic_curve::{group::cofactor::CofactorCurveAffine, Group};
use ff::{
    derive::bitvec::{order::Lsb0, view::AsBits},
    PrimeField,
};
use itertools::Itertools;
use snark_verifier_sdk::snark_verifier::{
    halo2_base::{
        halo2_proofs::halo2curves::secp256k1::{Fq, Secp256k1Affine},
        utils::log2_ceil,
    },
    util::arithmetic::CurveAffine,
};

// Note that the sqrt is at most 16, so this is a very dumb implementation.
pub fn sqrt_ceil(n: usize) -> usize {
    assert!(n <= 256);
    for i in 1..16 {
        if i * i >= n {
            return i;
        }
    }
    16
}

pub fn msm_axvm(g: Vec<Secp256k1Affine>, scalars: Vec<Fq>) -> Secp256k1Affine {
    let mut acc = Secp256k1Affine::identity();
    msm_serial::<Secp256k1Affine>(&scalars, &g, &mut acc);
    acc
}

pub fn msm_axvm_C<C: CurveAffine>(g: Vec<C>, scalars: Vec<C::Scalar>) -> C {
    let mut acc = C::identity();
    msm_serial::<C>(&scalars, &g, &mut acc);
    acc
}

pub fn multi_scalar_multiply(g: Vec<Secp256k1Affine>, scalars: Vec<Fq>) -> Secp256k1Affine {
    let n = g.len();
    let lambda = 256;
    let s = sqrt_ceil(lambda / n);
    let t = (lambda + s - 1) / s;
    let b = log2_ceil(t as u64) - log2_ceil(log2_ceil(t as u64) as u64);
    println!("s: {}, t: {}, b: {}", s, t, b);
    let e = scalars
        .iter()
        .map(|s| scalar_to_bits::<Secp256k1Affine>(*s))
        .collect_vec();
    let (g, e) = multi_scalar_multiply_internal0(g, e, s, t, lambda);
    let g = multi_scalar_multiply_internal1(g, e, b);
    multi_scalar_multiply_internal2(g, s)
}

// pub fn scalar_to_bits(scalar: Fq) -> Vec<bool> {
//     let scalar = scalar.to_bytes_le();
//     let mut bits = Vec::with_capacity(256);
//     for b in scalar {
//         let mut i = b;
//         for _ in 0..8 {
//             bits.push(i & 1 == 1);
//             i /= 2;
//         }
//     }
//     bits
// }

pub fn scalar_to_bits<C: CurveAffine>(scalar: C::Scalar) -> Vec<bool> {
    let scalar = scalar.to_repr();
    let scalar = scalar.as_bits::<Lsb0>();
    let mut bits = Vec::with_capacity(C::Scalar::NUM_BITS as usize);
    for b in scalar {
        // let mut i = b;
        // for _ in 0..8 {
        //     bits.push(i & 1 == 1);
        //     i /= 2;
        // }
        bits.push(*b);
    }
    bits
}

pub fn scalar_to_bytes<C: CurveAffine>(scalar: C::Scalar) -> Vec<u8> {
    let scalar = scalar_to_bits::<C>(scalar);
    let mut bytes = vec![0; C::Scalar::NUM_BITS.div_ceil(8u32) as usize];
    for (i, b) in scalar.iter().enumerate() {
        if *b {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

/// Performs a multi-scalar multiplication operation.
///
/// This function will panic if coeffs and bases have a different length.
pub fn msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| scalar_to_bytes::<C>(*a)).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    let field_byte_size = C::Scalar::NUM_BITS.div_ceil(8u32) as usize;
    // OR all coefficients in order to make a mask to figure out the maximum number of bytes used
    // among all coefficients.
    let mut acc_or = vec![0; field_byte_size];
    for coeff in &coeffs {
        for (acc_limb, limb) in acc_or.iter_mut().zip(coeff.iter()) {
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
        return;
    }
    let number_of_windows = max_byte_size * 8 / c + 1;

    for current_window in (0..number_of_windows).rev() {
        for _ in 0..c {
            *acc = ec_add::<C>(*acc, *acc);
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            // Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Affine(ec_add(a, *other)),
                    // Bucket::Projective(mut a) => {
                    //     a += *other;
                    //     Bucket::Projective(a)
                    // }
                }
            }

            fn add(self, mut other: C) -> C {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other = ec_add(a, other);
                        other
                    } // Bucket::Projective(a) => other + a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; 1 << (c - 1)];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_booth_index(current_window, c, coeff.as_ref());
            if coeff.is_positive() {
                buckets[coeff as usize - 1].add_assign(base);
            }
            if coeff.is_negative() {
                buckets[coeff.unsigned_abs() as usize - 1].add_assign(&base.neg());
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity().into();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc = ec_add(*acc, running_sum);
        }
    }
}

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

fn multi_scalar_multiply_internal0(
    g: Vec<Secp256k1Affine>,
    e: Vec<Vec<bool>>,
    s: usize,
    t: usize,
    lambda: usize,
) -> (Vec<Secp256k1Affine>, Vec<Vec<bool>>) {
    let n = g.len();
    let mut new_g = Vec::with_capacity(n * s);
    let mut new_e = vec![Vec::with_capacity(n * s); t];
    for (k, ne) in new_e.iter_mut().enumerate() {
        for (i, g) in g.iter().enumerate() {
            if k == 0 {
                new_g.push(*g);
                for _ in 1..s {
                    let last = new_g[new_g.len() - 1];
                    let double = ec_add(last, last);
                    new_g.push(double);
                }
            }
            let fill = if k == t - 1 { lambda % s } else { s };
            for j in 0..fill {
                ne.push(e[i][k * s + j]);
            }
            for _ in fill..s {
                ne.push(false);
            }
        }
    }
    (new_g, new_e)
}

fn multi_scalar_multiply_internal1(
    g: Vec<Secp256k1Affine>,
    e: Vec<Vec<bool>>,
    b: usize,
) -> Vec<Secp256k1Affine> {
    // let m = g.len();
    let s_chunks = g.chunks(b).collect::<Vec<_>>();
    let e_chunks = e.iter().map(|e| e.chunks(b).collect_vec()).collect_vec();
    let t = s_chunks
        .iter()
        .map(|s| {
            let iters = 1 << s.len();
            let mut prods = vec![];
            let mut cur_pow = 1;
            let mut cur_idx = 0;
            for i in 1..iters {
                if i == cur_pow * 2 {
                    cur_pow *= 2;
                    cur_idx += 1;
                }
                if i == cur_pow {
                    prods.push(s[cur_idx])
                } else {
                    prods.push((s[cur_idx] + prods[i - cur_pow - 1]).into());
                }
            }
            prods
        })
        .collect_vec();
    let g = e_chunks
        .iter()
        .map(|e| {
            let id = Secp256k1Affine::identity();
            e.iter().zip(t.iter()).fold(id, |acc, (e, t)| {
                let mut idx = 0;
                let l = e.len();
                for i in 0..l {
                    if i > 0 {
                        idx *= 2;
                    }
                    if e[l - i - 1] {
                        idx += 1;
                    }
                }
                if idx == 0 {
                    acc
                } else {
                    ec_add(acc, t[idx - 1])
                }
            })
        })
        .collect_vec();
    g
}

fn multi_scalar_multiply_internal2(g: Vec<Secp256k1Affine>, s: usize) -> Secp256k1Affine {
    let l = g.len();
    let mut acc = g[l - 1];
    for i in 0..l - 1 {
        for _ in 0..s {
            acc = ec_add(acc, acc);
        }
        acc = ec_add(acc, g[l - i - 2]);
    }
    acc
}

pub fn ec_add<C: CurveAffine>(g: C, h: C) -> C {
    let r = g + h;
    r.into()
}
