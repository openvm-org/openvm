use std::{collections::VecDeque, sync::Arc};

use num_bigint_dig::BigUint;

use crate::{
    modular_multiplication::{
        columns::{
            ModularMultiplicationAuxCols, ModularMultiplicationCols, ModularMultiplicationIoCols,
        },
        FullLimbs, LimbDimensions,
    },
    var_range::VariableRangeCheckerChip,
};

pub(super) fn big_uint_to_bits(x: BigUint) -> VecDeque<usize> {
    let mut result = VecDeque::new();
    for byte in x.to_bytes_le() {
        for i in 0..8 {
            result.push_back(((byte >> i) as usize) & 1);
        }
    }
    result
}

pub(super) fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    if limb_size == 0 {
        0
    } else {
        let bit = deque.pop_front().unwrap_or(0);
        bit + (2 * take_limb(deque, limb_size - 1))
    }
}

fn without_first_limbs<T: Clone>(limbs: &[Vec<T>]) -> Vec<Vec<T>> {
    limbs
        .iter()
        .map(|limbs_here| limbs_here[1..].to_vec())
        .collect()
}

pub fn generate_modular_multiplication_trace_row(
    modulus: BigUint,
    limb_dimensions: &LimbDimensions,
    range_checker: Arc<VariableRangeCheckerChip>,
    a: BigUint,
    b: BigUint,
) -> (ModularMultiplicationCols<usize>, FullLimbs<usize>) {
    let product = a.clone() * b.clone();
    let r = product.clone() % modulus.clone();
    let q = product.clone() / modulus.clone();

    let mut a_bits = big_uint_to_bits(a);
    let mut b_bits = big_uint_to_bits(b);
    let mut r_bits = big_uint_to_bits(r);
    let mut q_bits = big_uint_to_bits(q);

    let [(a_elems, a_limbs), (b_elems, b_limbs), (r_elems, r_limbs)] =
        [&mut a_bits, &mut b_bits, &mut r_bits].map(|bits| {
            let elems: (Vec<_>, Vec<_>) = limb_dimensions
                .io_limb_sizes
                .iter()
                .map(|limb_sizes_here| {
                    let mut elem = 0;
                    let mut shift = 0;
                    let limbs = limb_sizes_here
                        .iter()
                        .map(|&limb_size| {
                            let limb = take_limb(bits, limb_size);
                            range_checker.add_count(limb as u32, limb_size);
                            elem += limb << shift;
                            shift += limb_size;
                            limb
                        })
                        .collect();
                    (elem, limbs)
                })
                .unzip();
            assert!(bits.is_empty());
            elems
        });

    let q_limbs: Vec<usize> = limb_dimensions
        .q_limb_sizes
        .iter()
        .map(|&limb_size| {
            let limb = take_limb(&mut q_bits, limb_size);
            range_checker.add_count(limb as u32, limb_size);
            limb
        })
        .collect();
    assert!(q_bits.is_empty());

    let cols = ModularMultiplicationCols {
        io: ModularMultiplicationIoCols {
            a_elems,
            b_elems,
            r_elems,
        },
        aux: ModularMultiplicationAuxCols {
            a_limbs_without_first: without_first_limbs(&a_limbs),
            b_limbs_without_first: without_first_limbs(&b_limbs),
            r_limbs_without_first: without_first_limbs(&r_limbs),
            q_limbs: q_limbs.clone(),
        },
    };
    let full_limbs = FullLimbs {
        a_limbs,
        b_limbs,
        r_limbs,
        q_limbs,
    };
    (cols, full_limbs)
}
