use std::array::from_fn;

use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;

use crate::circuit::deferral::{DEF_INTERNAL_TAG, DEF_LEAF_TAG};

pub(crate) fn def_tagged_compress(
    tag: [u8; DIGEST_SIZE],
    left: [F; DIGEST_SIZE],
    right: [F; DIGEST_SIZE],
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    let tagged_left = poseidon2_compress_with_capacity(tag.map(F::from_u8), left).0;
    let parent = poseidon2_compress_with_capacity(tagged_left, right).0;
    (tagged_left, parent)
}

pub fn def_leaf_compress(
    left: [F; DIGEST_SIZE],
    right: [F; DIGEST_SIZE],
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    def_tagged_compress(DEF_LEAF_TAG, left, right)
}

pub(crate) fn def_internal_compress(
    left: [F; DIGEST_SIZE],
    right: [F; DIGEST_SIZE],
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    def_tagged_compress(DEF_INTERNAL_TAG, left, right)
}

pub(crate) fn def_zero_hash(depth: usize) -> [F; DIGEST_SIZE] {
    let mut zero_hash = [F::ZERO; DIGEST_SIZE];
    for level in 0..depth {
        zero_hash = if level == 0 {
            def_leaf_compress(zero_hash, zero_hash).1
        } else {
            def_internal_compress(zero_hash, zero_hash).1
        };
    }
    zero_hash
}

pub(crate) fn def_zero_hashes_from_depth_one<const MAX_DEPTH: usize>(
) -> [[F; DIGEST_SIZE]; MAX_DEPTH] {
    let mut zero_hash = [F::ZERO; DIGEST_SIZE];
    from_fn(|depth| {
        zero_hash = if depth == 0 {
            def_leaf_compress(zero_hash, zero_hash).1
        } else {
            def_internal_compress(zero_hash, zero_hash).1
        };
        zero_hash
    })
}
