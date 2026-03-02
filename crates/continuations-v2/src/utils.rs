use std::array::from_fn;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;

pub fn digests_to_poseidon2_input<T: Clone>(
    x: [T; DIGEST_SIZE],
    y: [T; DIGEST_SIZE],
) -> [T; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < DIGEST_SIZE {
            x[i].clone()
        } else {
            y[i - DIGEST_SIZE].clone()
        }
    })
}

pub fn poseidon2_input_to_digests<T>(
    x: [T; POSEIDON2_WIDTH],
) -> ([T; DIGEST_SIZE], [T; DIGEST_SIZE]) {
    let mut it = x.into_iter();
    let commit = from_fn(|_| it.next().unwrap());
    let len = from_fn(|_| it.next().unwrap());
    (commit, len)
}

pub fn pad_slice_to_poseidon2_input<T: Clone>(x: &[T], fill: T) -> [T; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < x.len() {
            x[i].clone()
        } else {
            fill.clone()
        }
    })
}

pub fn zero_hash(depth: usize) -> [F; DIGEST_SIZE] {
    let mut ret = [F::ZERO; DIGEST_SIZE];
    for _ in 0..depth {
        ret = poseidon2_compress_with_capacity(ret, ret).0;
    }
    ret
}
