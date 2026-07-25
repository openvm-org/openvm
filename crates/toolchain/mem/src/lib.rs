//! libc memory intrinsics for OpenVM guest programs.
//!
//! OpenVM costs a misaligned load or store the same as an aligned one, and the guest target
//! enables `+unaligned-scalar-mem`. A conventional libc spends most of its complexity avoiding
//! misaligned access, and all of that is wasted work here, so these implementations do no
//! alignment work at all: they issue 8-byte accesses at whatever address they are given and cover
//! short runs with overlapping moves rather than byte loops.
//!
//! `#![no_builtins]` keeps LLVM's loop-idiom pass from lowering the loops here back into calls to
//! the very symbols they define. It is scoped to this crate so the rest of the guest still gets
//! normal libcall recognition.

#![no_std]
#![no_builtins]

/// Bytes moved per iteration of the bulk loops.
const BLOCK: usize = 64;

#[inline(always)]
unsafe fn load<const N: usize>(src: *const u8) -> [u8; N] {
    (src as *const [u8; N]).read()
}

#[inline(always)]
unsafe fn store<const N: usize>(dest: *mut u8, val: [u8; N]) {
    (dest as *mut [u8; N]).write(val)
}

/// Scalar accessors. Values are kept in registers, unlike byte arrays wide enough to spill.
macro_rules! scalar_accessors {
    ($read:ident, $write:ident, $t:ty) => {
        #[inline(always)]
        unsafe fn $read(src: *const u8) -> $t {
            <$t>::from_ne_bytes(load::<{ core::mem::size_of::<$t>() }>(src))
        }
        #[inline(always)]
        unsafe fn $write(dest: *mut u8, val: $t) {
            store::<{ core::mem::size_of::<$t>() }>(dest, val.to_ne_bytes())
        }
    };
}
scalar_accessors!(read_u64, write_u64, u64);
scalar_accessors!(read_u32, write_u32, u32);
scalar_accessors!(read_u16, write_u16, u16);

/// Copies the first and last `WORDS * 8` bytes of an `n`-byte range, which together cover it when
/// `n <= WORDS * 16`.
///
/// Every load is issued before any store, so this stays correct when the ranges overlap: the two
/// runs together read exactly the bytes they go on to write. `WORDS` is a constant, so the loops
/// unroll and the values stay in registers.
#[inline(always)]
unsafe fn copy_overlapping<const WORDS: usize>(dest: *mut u8, src: *const u8, n: usize) {
    let back = n - WORDS * 8;
    let mut head = [0u64; WORDS];
    let mut tail = [0u64; WORDS];
    let mut i = 0;
    while i < WORDS {
        head[i] = read_u64(src.add(i * 8));
        tail[i] = read_u64(src.add(back + i * 8));
        i += 1;
    }
    i = 0;
    while i < WORDS {
        write_u64(dest.add(i * 8), head[i]);
        write_u64(dest.add(back + i * 8), tail[i]);
        i += 1;
    }
}

/// Copies `n < BLOCK` bytes using two overlapping runs of same-width moves.
#[inline(always)]
unsafe fn copy_tail(dest: *mut u8, src: *const u8, n: usize) {
    if n >= 32 {
        copy_overlapping::<4>(dest, src, n);
    } else if n >= 16 {
        copy_overlapping::<2>(dest, src, n);
    } else if n >= 8 {
        copy_overlapping::<1>(dest, src, n);
    } else if n >= 4 {
        let (a, b) = (read_u32(src), read_u32(src.add(n - 4)));
        write_u32(dest, a);
        write_u32(dest.add(n - 4), b);
    } else if n >= 2 {
        let (a, b) = (read_u16(src), read_u16(src.add(n - 2)));
        write_u16(dest, a);
        write_u16(dest.add(n - 2), b);
    } else if n == 1 {
        *dest = *src;
    }
}

/// Copies low addresses first.
///
/// # Safety
///
/// `src` must be valid for reads of `n` bytes and `dest` valid for writes of `n` bytes. The
/// ranges may overlap only when `dest <= src`.
#[inline(always)]
pub unsafe fn copy_forward(dest: *mut u8, src: *const u8, n: usize) {
    let (mut dest, mut src, mut rem) = (dest, src, n);
    while rem >= BLOCK {
        store::<BLOCK>(dest, load::<BLOCK>(src));
        dest = dest.add(BLOCK);
        src = src.add(BLOCK);
        rem -= BLOCK;
    }
    copy_tail(dest, src, rem);
}

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    copy_forward(dest, src, n);
    dest
}

/// Writes `WORDS` words at `dest` and `WORDS` more ending at `dest + n`.
#[inline(always)]
unsafe fn set_overlapping<const WORDS: usize>(dest: *mut u8, word: u64, n: usize) {
    let back = n - WORDS * 8;
    let mut i = 0;
    while i < WORDS {
        write_u64(dest.add(i * 8), word);
        write_u64(dest.add(back + i * 8), word);
        i += 1;
    }
}

/// Writes `n < BLOCK` copies of `word`'s low byte using two overlapping runs of stores.
///
/// Overlapping stores of the same value are idempotent, so no ordering care is needed.
#[inline(always)]
unsafe fn set_tail(dest: *mut u8, word: u64, n: usize) {
    if n >= 32 {
        set_overlapping::<4>(dest, word, n);
    } else if n >= 16 {
        set_overlapping::<2>(dest, word, n);
    } else if n >= 8 {
        set_overlapping::<1>(dest, word, n);
    } else if n >= 4 {
        write_u32(dest, word as u32);
        write_u32(dest.add(n - 4), word as u32);
    } else if n >= 2 {
        write_u16(dest, word as u16);
        write_u16(dest.add(n - 2), word as u16);
    } else if n == 1 {
        *dest = word as u8;
    }
}

/// Fills `n` bytes with `val`.
///
/// # Safety
///
/// `dest` must be valid for writes of `n` bytes.
#[inline(always)]
pub unsafe fn set_bytes(dest: *mut u8, val: u8, n: usize) {
    let word = u64::from_ne_bytes([val; 8]);
    let (mut dest, mut rem) = (dest, n);
    while rem >= BLOCK {
        let mut i = 0;
        while i < BLOCK / 8 {
            write_u64(dest.add(i * 8), word);
            i += 1;
        }
        dest = dest.add(BLOCK);
        rem -= BLOCK;
    }
    set_tail(dest, word, rem);
}

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
pub unsafe extern "C" fn memset(dest: *mut u8, val: core::ffi::c_int, n: usize) -> *mut u8 {
    set_bytes(dest, val as u8, n);
    dest
}

#[cfg(test)]
mod tests;
