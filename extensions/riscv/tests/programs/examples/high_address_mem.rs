#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::{arch::asm, hint::black_box, ptr};

openvm::entry!(main);

/// End of the pre-2^32 guest platform memory (512 MiB); kept as a representative
/// high-address boundary now that `openvm_platform::memory::MEM_SIZE` spans the full
/// 2^32 bytes.
const LEGACY_PLATFORM_MEM_END: u64 = 1 << 29;
/// One past the last addressable byte (`openvm_platform::memory::MEM_SIZE`).
const MEM_TOP: u64 = 1 << 32;

fn addr<T>(byte_addr: u64) -> *mut T {
    black_box(byte_addr as usize) as *mut T
}

unsafe fn round_trip<T: Copy + PartialEq + core::fmt::Debug>(byte_addr: u64, value: T) {
    let p = addr::<T>(byte_addr);
    ptr::write_volatile(p, value);
    assert_eq!(ptr::read_volatile(p), value, "round trip at {byte_addr:#x}");
}

pub fn main() {
    unsafe {
        // Aligned round trips of every access width, from just past the legacy
        // platform memory bound up through the 2^31 cell-pointer top bit.
        round_trip::<u64>(LEGACY_PLATFORM_MEM_END, 0x0123_4567_89ab_cdef);
        round_trip::<u32>((1 << 30) + 0x40, 0xdead_beef);
        round_trip::<u16>((1 << 31) + 0x10, 0xa55a);
        round_trip::<u8>(0xc000_0003, 0x5a);

        // The last addressable 8-byte block starts untouched (zero-initialized)...
        let top_dword = addr::<u64>(MEM_TOP - 8);
        assert_eq!(ptr::read_volatile(top_dword), 0);
        ptr::write_volatile(top_dword, 0x1122_3344_5566_7788);
        // ... and narrow writes at the very top of memory merge into it.
        ptr::write_volatile(addr::<u32>(MEM_TOP - 4), 0x99aa_bbcc);
        ptr::write_volatile(addr::<u16>(MEM_TOP - 2), 0xddee);
        ptr::write_volatile(addr::<u8>(MEM_TOP - 1), 0xff);
        assert_eq!(ptr::read_volatile(top_dword), 0xffee_bbcc_5566_7788);

        // Reads of untouched high pages see zero-initialized memory.
        assert_eq!(ptr::read_volatile(addr::<u64>(0xf123_4560)), 0);

        // Loads at high addresses still sign- and zero-extend correctly.
        ptr::write_volatile(addr::<u8>(0x8000_0100), 0x80);
        assert_eq!(ptr::read_volatile(addr::<i8>(0x8000_0100)), i8::MIN);
        ptr::write_volatile(addr::<u16>(0x8000_0102), 0x8001);
        assert_eq!(ptr::read_volatile(addr::<i16>(0x8000_0102)), -0x7fff);
        ptr::write_volatile(addr::<u32>(0x8000_0104), 0x8000_0001);
        assert_eq!(ptr::read_volatile(addr::<i32>(0x8000_0104)), -0x7fff_ffff);

        let mut val: u64;
        // A misaligned store crossing the 2^31 boundary writes into both blocks.
        asm!(
            "sd {v}, 0({p})",
            v = in(reg) 0x1122_3344_5566_7788u64,
            p = in(reg) black_box(0x7fff_fffcu64),
        );
        assert_eq!(ptr::read_volatile(addr::<u32>(0x7fff_fffc)), 0x5566_7788);
        assert_eq!(ptr::read_volatile(addr::<u32>(0x8000_0000)), 0x1122_3344);

        // rs1 + imm carries through the low 16-bit pointer limb and lands on 2^31.
        asm!("lwu {v}, 7({p})", v = out(reg) val, p = in(reg) black_box(0x7fff_fff9u64));
        assert_eq!(val, 0x1122_3344);

        // A misaligned load whose second block address carries across a 16-bit limb.
        ptr::write_volatile(addr::<u64>(0x8000_fff8), 0x0807_0605_0403_0201);
        ptr::write_volatile(addr::<u64>(0x8001_0000), 0x100f_0e0d_0c0b_0a09);
        asm!("ld {v}, 0({p})", v = out(reg) val, p = in(reg) black_box(0x8000_fffcu64));
        assert_eq!(val, 0x0c0b_0a09_0807_0605);

        // A misaligned load spanning the last two blocks of memory.
        ptr::write_volatile(addr::<u64>(0xffff_fff0), 0xf7f6_f5f4_f3f2_f1f0);
        asm!("ld {v}, 0({p})", v = out(reg) val, p = in(reg) black_box(0xffff_fff1u64));
        assert_eq!(val, 0x88f7_f6f5_f4f3_f2f1);

        // The maximum immediate offset reaches the last addressable byte.
        asm!("lbu {v}, 2047({p})", v = out(reg) val, p = in(reg) black_box(0xffff_f800u64));
        assert_eq!(val, 0xff);

        // Bulk memory intrinsics work on high pages: copy across the 3 GiB page
        // boundary, then memset over it.
        let src: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
        let dst = addr::<u8>(0xbfff_fff0);
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        for (i, &expected) in src.iter().enumerate() {
            assert_eq!(ptr::read_volatile(dst.add(i)), expected);
        }
        ptr::write_bytes(dst, 0xa5, src.len());
        for i in 0..src.len() {
            assert_eq!(ptr::read_volatile(dst.add(i)), 0xa5);
        }
    }
}
