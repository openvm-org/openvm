#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

#[cfg(target_arch = "riscv64")]
mod riscv64 {
    use core::arch::asm;

    macro_rules! rr {
        ($name:ident, $mnemonic:literal) => {
            fn $name(a: u64, b: u64) -> u64 {
                let out: u64;
                unsafe {
                    asm!(
                        concat!($mnemonic, " {out}, {a}, {b}"),
                        out = lateout(reg) out,
                        a = in(reg) a,
                        b = in(reg) b,
                        options(nomem, nostack)
                    );
                }
                out
            }
        };
    }

    macro_rules! ri {
        ($name:ident, $mnemonic:literal, $imm:literal) => {
            fn $name(a: u64) -> u64 {
                let out: u64;
                unsafe {
                    asm!(
                        concat!($mnemonic, " {out}, {a}, ", $imm),
                        out = lateout(reg) out,
                        a = in(reg) a,
                        options(nomem, nostack)
                    );
                }
                out
            }
        };
    }

    macro_rules! unary {
        ($name:ident, $mnemonic:literal) => {
            fn $name(a: u64) -> u64 {
                let out: u64;
                unsafe {
                    asm!(
                        concat!($mnemonic, " {out}, {a}"),
                        out = lateout(reg) out,
                        a = in(reg) a,
                        options(nomem, nostack)
                    );
                }
                out
            }
        };
    }

    rr!(add_uw, "add.uw");
    rr!(sh1add, "sh1add");
    rr!(sh2add, "sh2add");
    rr!(sh3add, "sh3add");
    rr!(sh1add_uw, "sh1add.uw");
    rr!(sh2add_uw, "sh2add.uw");
    rr!(sh3add_uw, "sh3add.uw");
    ri!(slli_uw, "slli.uw", "13");

    rr!(andn, "andn");
    rr!(orn, "orn");
    rr!(xnor, "xnor");
    unary!(clz, "clz");
    unary!(ctz, "ctz");
    unary!(cpop, "cpop");
    unary!(clzw, "clzw");
    unary!(ctzw, "ctzw");
    unary!(cpopw, "cpopw");
    rr!(min, "min");
    rr!(minu, "minu");
    rr!(max, "max");
    rr!(maxu, "maxu");
    unary!(sext_b, "sext.b");
    unary!(sext_h, "sext.h");
    unary!(zext_h, "zext.h");
    rr!(rol, "rol");
    rr!(ror, "ror");
    ri!(rori, "rori", "13");
    rr!(rolw, "rolw");
    rr!(rorw, "rorw");
    ri!(roriw, "roriw", "13");
    unary!(orc_b, "orc.b");
    unary!(rev8, "rev8");

    rr!(bclr, "bclr");
    rr!(bset, "bset");
    rr!(binv, "binv");
    rr!(bext, "bext");
    ri!(bclri, "bclri", "47");
    ri!(bseti, "bseti", "48");
    ri!(binvi, "binvi", "49");
    ri!(bexti, "bexti", "50");

    fn check(actual: u64, expected: u64) {
        if actual != expected {
            openvm::process::panic();
        }
    }

    fn sext32(x: u32) -> u64 {
        (x as i32 as i64) as u64
    }

    fn orc_b_expected(x: u64) -> u64 {
        let mut out = 0;
        let mut i = 0;
        while i < 8 {
            if ((x >> (i * 8)) & 0xff) != 0 {
                out |= 0xffu64 << (i * 8);
            }
            i += 1;
        }
        out
    }

    pub fn run() {
        let x = core::hint::black_box(0x8123_4567_89ab_cdefu64);
        let y = core::hint::black_box(0x7654_3210_fedc_ba91u64);
        let u = core::hint::black_box(0x00f0_0000_1000_0000u64);
        let shamt64 = (y & 63) as u32;
        let shamt32 = (y & 31) as u32;

        check(add_uw(x, y), (x as u32 as u64).wrapping_add(y));
        check(sh1add(x, y), x.wrapping_shl(1).wrapping_add(y));
        check(sh2add(x, y), x.wrapping_shl(2).wrapping_add(y));
        check(sh3add(x, y), x.wrapping_shl(3).wrapping_add(y));
        check(sh1add_uw(x, y), ((x as u32 as u64) << 1).wrapping_add(y));
        check(sh2add_uw(x, y), ((x as u32 as u64) << 2).wrapping_add(y));
        check(sh3add_uw(x, y), ((x as u32 as u64) << 3).wrapping_add(y));
        check(slli_uw(x), (x as u32 as u64) << 13);

        check(andn(x, y), x & !y);
        check(orn(x, y), x | !y);
        check(xnor(x, y), !(x ^ y));
        check(clz(x), x.leading_zeros() as u64);
        check(ctz(x), x.trailing_zeros() as u64);
        check(cpop(x), x.count_ones() as u64);
        check(clzw(x), (x as u32).leading_zeros() as u64);
        check(ctzw(x), (x as u32).trailing_zeros() as u64);
        check(cpopw(x), (x as u32).count_ones() as u64);
        check(min(x, y), core::cmp::min(x as i64, y as i64) as u64);
        check(minu(x, y), core::cmp::min(x, y));
        check(max(x, y), core::cmp::max(x as i64, y as i64) as u64);
        check(maxu(x, y), core::cmp::max(x, y));
        check(sext_b(x), (x as i8 as i64) as u64);
        check(sext_h(x), (x as i16 as i64) as u64);
        check(zext_h(x), x & 0xffff);
        check(rol(x, y), x.rotate_left(shamt64));
        check(ror(x, y), x.rotate_right(shamt64));
        check(rori(x), x.rotate_right(13));
        check(rolw(x, y), sext32((x as u32).rotate_left(shamt32)));
        check(rorw(x, y), sext32((x as u32).rotate_right(shamt32)));
        check(roriw(x), sext32((x as u32).rotate_right(13)));
        check(orc_b(u), orc_b_expected(u));
        check(rev8(x), x.swap_bytes());

        check(bclr(x, y), x & !(1u64 << shamt64));
        check(bset(x, y), x | (1u64 << shamt64));
        check(binv(x, y), x ^ (1u64 << shamt64));
        check(bext(x, y), (x >> shamt64) & 1);
        check(bclri(x), x & !(1u64 << 47));
        check(bseti(x), x | (1u64 << 48));
        check(binvi(x), x ^ (1u64 << 49));
        check(bexti(x), (x >> 50) & 1);
    }
}

#[cfg(target_arch = "riscv64")]
pub fn main() {
    riscv64::run();
}

#[cfg(not(target_arch = "riscv64"))]
pub fn main() {}
