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
fn compare_bytes_matches_slice_cmp() {
    let a = pattern(MAX_LEN + 2 * OFFSETS);
    for n in 0..=MAX_LEN {
        for a_off in 0..OFFSETS {
            for b_off in 0..OFFSETS {
                // Flip one byte at a time so every position is the first difference in turn,
                // including positions only the overlapping final word can reach.
                for flip in (0..n).chain([usize::MAX]) {
                    let mut b = a.clone();
                    if flip != usize::MAX {
                        b[b_off + flip] ^= 0x80;
                    }
                    let got =
                        unsafe { compare_bytes(a.as_ptr().add(a_off), b.as_ptr().add(b_off), n) };
                    let want = a[a_off..a_off + n].cmp(&b[b_off..b_off + n]);
                    assert_eq!(
                        got.signum(),
                        match want {
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                            std::cmp::Ordering::Greater => 1,
                        },
                        "n={n} a_off={a_off} b_off={b_off} flip={flip}"
                    );
                    let differ =
                        unsafe { bytes_differ(a.as_ptr().add(a_off), b.as_ptr().add(b_off), n) };
                    assert_eq!(differ, want != std::cmp::Ordering::Equal);
                }
            }
        }
    }
}

/// Overlap distances that straddle the bulk-loop stride in both copy directions.
const MAX_DELTA: usize = BLOCK + OFFSETS;

#[test]
fn move_bytes_matches_copy_within() {
    let base = pattern(MAX_LEN + 2 * MAX_DELTA);
    let mut buf = base.clone();
    let mut expected = base.clone();
    for n in 0..=MAX_LEN {
        for delta in 0..=MAX_DELTA {
            // `dest` above `src` forces the backward copy, below it the forward one.
            for (src_off, dest_off) in [
                (MAX_DELTA, MAX_DELTA + delta),
                (MAX_DELTA + delta, MAX_DELTA),
            ] {
                buf.copy_from_slice(&base);
                expected.copy_from_slice(&base);
                expected.copy_within(src_off..src_off + n, dest_off);
                unsafe { move_bytes(buf.as_mut_ptr().add(dest_off), buf.as_ptr().add(src_off), n) };
                assert_eq!(buf, expected, "n={n} delta={delta} dest_off={dest_off}");
            }
        }
    }
}

#[test]
fn set_bytes_matches_fill() {
    for n in 0..=MAX_LEN {
        for dest_off in 0..OFFSETS {
            let mut dest = vec![PAD; MAX_LEN + 2 * OFFSETS];
            unsafe { set_bytes(dest.as_mut_ptr().add(dest_off), 0x5C, n) };
            assert!(
                dest[dest_off..dest_off + n].iter().all(|&b| b == 0x5C),
                "n={n} dest_off={dest_off}"
            );
            assert!(
                dest[..dest_off].iter().all(|&b| b == PAD)
                    && dest[dest_off + n..].iter().all(|&b| b == PAD),
                "wrote outside the destination range: n={n} dest_off={dest_off}"
            );
        }
    }
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
