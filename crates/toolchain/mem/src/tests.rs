extern crate std;

use std::{vec, vec::Vec};

use super::*;

const PAD: u8 = 0xAA;
/// Every source/destination offset within a memory block, plus one past it.
const OFFSETS: usize = 9;
/// Exercises both bulk-loop iterations and every `copy_tail` arm.
const MAX_LEN: usize = 200;

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect()
}

#[test]
fn copy_forward_matches_slice_copy() {
    let src = pattern(MAX_LEN + 2 * OFFSETS);
    for n in 0..=MAX_LEN {
        for src_off in 0..OFFSETS {
            for dest_off in 0..OFFSETS {
                let mut dest = vec![PAD; MAX_LEN + 2 * OFFSETS];
                unsafe {
                    copy_forward(
                        dest.as_mut_ptr().add(dest_off),
                        src.as_ptr().add(src_off),
                        n,
                    )
                };
                assert_eq!(
                    &dest[dest_off..dest_off + n],
                    &src[src_off..src_off + n],
                    "n={n} src_off={src_off} dest_off={dest_off}"
                );
                assert!(
                    dest[..dest_off].iter().all(|&b| b == PAD)
                        && dest[dest_off + n..].iter().all(|&b| b == PAD),
                    "wrote outside the destination range: n={n} dest_off={dest_off}"
                );
            }
        }
    }
}
