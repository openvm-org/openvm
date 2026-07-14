#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern "C" {
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

openvm::entry!(main);

// Covers the fast small-copy switch (n <= 32), the aligned word path, the
// shifted-word path for every src/dst alignment mismatch mod 8, and
// head/tail handling around the 32-byte and word boundaries.
const LENGTHS: [usize; 18] = [
    0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255,
];

const CANARY: u8 = 0xCC;

fn main() {
    let mut src = [0u8; 272];
    let mut dst = [0u8; 272];
    let mut i = 0;
    while i < src.len() {
        src[i] = (i as u8).wrapping_mul(31).wrapping_add(7);
        i += 1;
    }

    for src_off in 0..8 {
        for dst_off in 0..8 {
            for &n in LENGTHS.iter() {
                for b in dst.iter_mut() {
                    *b = CANARY;
                }
                let ret =
                    unsafe { memcpy(dst.as_mut_ptr().add(dst_off), src.as_ptr().add(src_off), n) };
                assert_eq!(ret as usize, dst.as_ptr() as usize + dst_off);
                for &b in dst.iter().take(dst_off) {
                    assert_eq!(b, CANARY);
                }
                for i in 0..n {
                    assert_eq!(dst[dst_off + i], src[src_off + i]);
                }
                for &b in dst.iter().skip(dst_off + n) {
                    assert_eq!(b, CANARY);
                }
            }
        }
    }
}
