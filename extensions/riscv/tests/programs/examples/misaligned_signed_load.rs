#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

/// 8-byte aligned buffer so the byte shifts exercised below are deterministic.
#[repr(align(8))]
struct Aligned([u8; 24]);

/// Mix of bytes with the high bit set and cleared so both sign-extension paths are hit.
fn init_bytes() -> [u8; 24] {
    let mut bytes = [0u8; 24];
    let mut i = 0;
    while i < bytes.len() {
        bytes[i] = (i as u8).wrapping_mul(37).wrapping_add(11);
        i += 1;
    }
    bytes
}

pub fn main() {
    let buf = Aligned(init_bytes());
    let model = init_bytes();
    let base = buf.0.as_ptr() as usize;

    let mut val: u64;
    unsafe {
        for shift in 0usize..8 {
            asm!("lb {v}, 0({p})", v = out(reg) val, p = in(reg) base + shift);
            assert_eq!(val, model[shift] as i8 as i64 as u64, "lb at shift {shift}");

            asm!("lh {v}, 0({p})", v = out(reg) val, p = in(reg) base + shift);
            let expected = i16::from_le_bytes([model[shift], model[shift + 1]]) as i64 as u64;
            assert_eq!(val, expected, "lh at shift {shift}");

            asm!("lw {v}, 0({p})", v = out(reg) val, p = in(reg) base + shift);
            let mut word = [0u8; 4];
            word.copy_from_slice(&model[shift..shift + 4]);
            let expected = i32::from_le_bytes(word) as i64 as u64;
            assert_eq!(val, expected, "lw at shift {shift}");
        }
    }
}
