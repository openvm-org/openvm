#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

/// 8-byte aligned buffer so the byte shifts exercised below are deterministic.
#[repr(align(8))]
struct Aligned([u8; 40]);

fn init_bytes() -> [u8; 40] {
    let mut bytes = [0u8; 40];
    let mut i = 0;
    while i < 40 {
        bytes[i] = (i as u8).wrapping_mul(37).wrapping_add(11);
        i += 1;
    }
    bytes
}

fn expect_u64(bytes: &[u8; 40], offset: usize) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(arr)
}

fn expect_u32(bytes: &[u8; 40], offset: usize) -> u64 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(arr) as u64
}

fn expect_u16(bytes: &[u8; 40], offset: usize) -> u64 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u64
}

pub fn main() {
    let mut buf = Aligned(init_bytes());
    let model = init_bytes();
    let base = buf.0.as_mut_ptr() as usize;

    // Misaligned (unsigned) loads via explicit instructions, covering both in-block
    // misalignment and accesses spanning two 8-byte blocks.
    let mut val: u64;
    unsafe {
        // ld: every nonzero shift spans two blocks.
        asm!("ld {v}, 3({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u64(&model, 3));
        asm!("ld {v}, 7({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u64(&model, 7));

        // lwu: shifts 1..=3 stay in one block, 5..=7 span two.
        asm!("lwu {v}, 1({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u32(&model, 1));
        asm!("lwu {v}, 6({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u32(&model, 6));

        // lhu: odd shifts are misaligned; shift 7 spans two blocks.
        asm!("lhu {v}, 5({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u16(&model, 5));
        asm!("lhu {v}, 15({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, expect_u16(&model, 15));

        // Sign-extending misaligned loads.
        asm!("lb {v}, 4({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, model[4] as i8 as i64 as u64);
        // lh at odd shift (in-block) and spanning two blocks.
        asm!("lh {v}, 3({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, i16::from_le_bytes([model[3], model[4]]) as i64 as u64);
        asm!("lh {v}, 7({p})", v = out(reg) val, p = in(reg) base);
        assert_eq!(val, i16::from_le_bytes([model[7], model[8]]) as i64 as u64);
        // lw misaligned in-block and spanning two blocks.
        asm!("lw {v}, 2({p})", v = out(reg) val, p = in(reg) base);
        let mut w = [0u8; 4];
        w.copy_from_slice(&model[2..6]);
        assert_eq!(val, i32::from_le_bytes(w) as i64 as u64);
        asm!("lw {v}, 5({p})", v = out(reg) val, p = in(reg) base);
        w.copy_from_slice(&model[5..9]);
        assert_eq!(val, i32::from_le_bytes(w) as i64 as u64);
    }

    // Misaligned stores: check the stored bytes land where they should and that every
    // other byte of the touched blocks is preserved (read-modify-write).
    let mut model = model;

    unsafe {
        let v: u64 = 0x1122_3344_5566_7788;
        // sd at offset 19: blocks at 16 and 24.
        asm!("sd {v}, 19({p})", v = in(reg) v, p = in(reg) base);
        model[19..27].copy_from_slice(&v.to_le_bytes());

        let v32: u32 = 0xa1b2_c3d4;
        // sw at offset 13: blocks at 8 and 16.
        asm!("sw {v}, 13({p})", v = in(reg) v32 as usize, p = in(reg) base);
        model[13..17].copy_from_slice(&v32.to_le_bytes());

        let v16: u16 = 0x9e8f;
        // sh at offset 33: odd in-block misalignment.
        asm!("sh {v}, 33({p})", v = in(reg) v16 as usize, p = in(reg) base);
        model[33..35].copy_from_slice(&v16.to_le_bytes());

        // sh at offset 31: blocks at 24 and 32.
        let v16b: u16 = 0x7654;
        asm!("sh {v}, 31({p})", v = in(reg) v16b as usize, p = in(reg) base);
        model[31..33].copy_from_slice(&v16b.to_le_bytes());
    }

    assert_eq!(&buf.0[..], &model[..]);
}
