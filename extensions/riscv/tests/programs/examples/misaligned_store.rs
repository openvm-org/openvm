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
    let mut buf = Aligned(init_bytes());
    let mut model = init_bytes();
    let base = buf.0.as_mut_ptr() as usize;

    unsafe {
        for shift in 0usize..8 {
            let v8 = 0xc0u8.wrapping_add(shift as u8);
            asm!("sb {v}, 0({p})", v = in(reg) v8 as usize, p = in(reg) base + shift);
            model[shift] = v8;

            let v16 = 0x9e00u16 + shift as u16;
            asm!("sh {v}, 0({p})", v = in(reg) v16 as usize, p = in(reg) base + shift);
            model[shift..shift + 2].copy_from_slice(&v16.to_le_bytes());

            let v32 = 0xa1b2_c300u32 + shift as u32;
            asm!("sw {v}, 0({p})", v = in(reg) v32 as usize, p = in(reg) base + shift);
            model[shift..shift + 4].copy_from_slice(&v32.to_le_bytes());

            let v64 = 0x1122_3344_5566_7700u64 + shift as u64;
            asm!("sd {v}, 0({p})", v = in(reg) v64 as usize, p = in(reg) base + shift);
            model[shift..shift + 8].copy_from_slice(&v64.to_le_bytes());
        }
    }

    // Every stored byte landed where it should, and every byte outside the stores
    // (including the untouched parts of the blocks) is preserved.
    assert_eq!(&buf.0[..], &model[..]);
}
