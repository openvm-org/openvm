#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

openvm::entry!(main);

/// Room for the longest case plus every source and destination offset.
const CAP: usize = 128;
const FILL: u8 = 0xAA;
/// Straddles the bulk-loop threshold and every overlapping-move arm of the tail.
const LENS: [usize; 13] = [0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64];
/// Every byte offset within an 8-byte memory block, plus one past it.
const OFFSETS: usize = 9;

fn byte(i: usize) -> u8 {
    (i as u8).wrapping_mul(31).wrapping_add(7)
}

fn check_memcpy(src: &[u8; CAP], dest: &mut [u8; CAP]) {
    for n in LENS {
        for src_off in 0..OFFSETS {
            for dest_off in 0..OFFSETS {
                *dest = [FILL; CAP];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        black_box(src.as_ptr().add(src_off)),
                        black_box(dest.as_mut_ptr().add(dest_off)),
                        black_box(n),
                    );
                }
                for k in 0..n {
                    assert_eq!(dest[dest_off + k], src[src_off + k]);
                }
                for k in 0..dest_off {
                    assert_eq!(dest[k], FILL);
                }
                for k in dest_off + n..CAP {
                    assert_eq!(dest[k], FILL);
                }
            }
        }
    }
}

fn check_memset(dest: &mut [u8; CAP]) {
    for n in LENS {
        for dest_off in 0..OFFSETS {
            *dest = [FILL; CAP];
            unsafe {
                core::ptr::write_bytes(
                    black_box(dest.as_mut_ptr().add(dest_off)),
                    black_box(0x5Cu8),
                    black_box(n),
                );
            }
            for k in 0..n {
                assert_eq!(dest[dest_off + k], 0x5C);
            }
            for k in 0..dest_off {
                assert_eq!(dest[k], FILL);
            }
            for k in dest_off + n..CAP {
                assert_eq!(dest[k], FILL);
            }
        }
    }
}

pub fn main() {
    let mut src = [0u8; CAP];
    for (i, b) in src.iter_mut().enumerate() {
        *b = byte(i);
    }
    let mut dest = [0u8; CAP];

    check_memcpy(&src, &mut dest);
    check_memset(&mut dest);
}
