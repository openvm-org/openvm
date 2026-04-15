//! FFI functions for the Int256 extension.
//!
//! These Rust functions are called from generated C code. They receive resolved
//! register values and the state as an opaque pointer. Memory reads/writes,
//! computation, and chip tracing go through double FFI; register access stays
//! on the C side.

use std::ffi::c_void;

use rvr_openvm_ext_ffi_common::{rd_mem_words_traced, wr_mem_words_traced, WORD_SIZE};

/// Number of bytes in a 256-bit integer.
const INT256_BYTES: usize = 32;

/// Number of 4-byte words in a 256-bit integer.
const INT256_WORDS: usize = INT256_BYTES / WORD_SIZE;

/// Bytes in one u64 limb.
const U64_LIMB_BYTES: usize = size_of::<u64>();

/// Number of 8-byte limbs in a 256-bit integer.
const INT256_U64_LIMBS: usize = INT256_BYTES / U64_LIMB_BYTES;

/// Bits per u64 limb.
const U64_BITS: u32 = u64::BITS;

/// Index of the most significant byte of a 256-bit value (sign byte for i256).
const INT256_MSB: usize = INT256_BYTES - 1;

/// Bit position of the sign within the most significant byte.
const SIGN_BIT_SHIFT: u32 = 7;

/// Mask of the sign bit within the most significant byte.
const SIGN_BIT_MASK: u8 = 1 << SIGN_BIT_SHIFT;

/// Read a 256-bit value from memory as INT256_WORDS LE words.
#[inline]
unsafe fn read_int256(state: *mut c_void, ptr: u32) -> [u8; INT256_BYTES] {
    let mut words = [0u32; INT256_WORDS];
    rd_mem_words_traced(state, ptr, &mut words);
    let mut bytes = [0u8; INT256_BYTES];
    for (i, w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Write a 256-bit value to memory as INT256_WORDS LE words.
#[inline]
unsafe fn write_int256(state: *mut c_void, ptr: u32, bytes: &[u8; INT256_BYTES]) {
    let mut words = [0u32; INT256_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    wr_mem_words_traced(state, ptr, &words);
}

// ── Byte-level helpers ──────────────────────────────────────────────────────

#[inline]
fn to_u64(b: &[u8; INT256_BYTES]) -> [u64; INT256_U64_LIMBS] {
    let mut out = [0u64; INT256_U64_LIMBS];
    for i in 0..INT256_U64_LIMBS {
        out[i] = u64::from_le_bytes(
            b[i * U64_LIMB_BYTES..(i + 1) * U64_LIMB_BYTES]
                .try_into()
                .unwrap(),
        );
    }
    out
}

#[inline]
fn from_u64(a: &[u64; INT256_U64_LIMBS]) -> [u8; INT256_BYTES] {
    let mut out = [0u8; INT256_BYTES];
    for i in 0..INT256_U64_LIMBS {
        out[i * U64_LIMB_BYTES..(i + 1) * U64_LIMB_BYTES].copy_from_slice(&a[i].to_le_bytes());
    }
    out
}

#[inline]
fn to_u32_arr(b: &[u8; INT256_BYTES]) -> [u32; INT256_WORDS] {
    let mut out = [0u32; INT256_WORDS];
    for i in 0..INT256_WORDS {
        out[i] = u32::from_le_bytes(b[i * WORD_SIZE..(i + 1) * WORD_SIZE].try_into().unwrap());
    }
    out
}

#[inline]
fn from_u32_arr(a: &[u32; INT256_WORDS]) -> [u8; INT256_BYTES] {
    let mut out = [0u8; INT256_BYTES];
    for i in 0..INT256_WORDS {
        out[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&a[i].to_le_bytes());
    }
    out
}

// ── Arithmetic ──────────────────────────────────────────────────────────────

fn u256_add(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let mut r = [0u64; INT256_U64_LIMBS];
    let (res, mut carry) = a[0].overflowing_add(b[0]);
    r[0] = res;
    for i in 1..INT256_U64_LIMBS {
        let (r1, c1) = a[i].overflowing_add(b[i]);
        let (r2, c2) = r1.overflowing_add(carry as u64);
        carry = c1 || c2;
        r[i] = r2;
    }
    from_u64(&r)
}

fn u256_sub(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let mut r = [0u64; INT256_U64_LIMBS];
    let (res, mut borrow) = a[0].overflowing_sub(b[0]);
    r[0] = res;
    for i in 1..INT256_U64_LIMBS {
        let (r1, c1) = a[i].overflowing_sub(b[i]);
        let (r2, c2) = r1.overflowing_sub(borrow as u64);
        borrow = c1 || c2;
        r[i] = r2;
    }
    from_u64(&r)
}

fn u256_xor(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let mut r = [0u64; INT256_U64_LIMBS];
    for i in 0..INT256_U64_LIMBS {
        r[i] = a[i] ^ b[i];
    }
    from_u64(&r)
}

fn u256_or(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let mut r = [0u64; INT256_U64_LIMBS];
    for i in 0..INT256_U64_LIMBS {
        r[i] = a[i] | b[i];
    }
    from_u64(&r)
}

fn u256_and(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let mut r = [0u64; INT256_U64_LIMBS];
    for i in 0..INT256_U64_LIMBS {
        r[i] = a[i] & b[i];
    }
    from_u64(&r)
}

fn u256_sll(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let shift = (b[0] & 0xff) as u32;
    let index_offset = (shift / U64_BITS) as usize;
    let bit_offset = shift % U64_BITS;
    let mut r = [0u64; INT256_U64_LIMBS];
    let mut carry = 0u64;
    for i in index_offset..INT256_U64_LIMBS {
        let curr = a[i - index_offset];
        r[i] = (curr << bit_offset) + carry;
        if bit_offset > 0 {
            carry = curr >> (U64_BITS - bit_offset);
        }
    }
    from_u64(&r)
}

fn u256_srl(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    shift_right(a, b, 0)
}

fn u256_sra(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    if a[INT256_MSB] & SIGN_BIT_MASK > 0 {
        shift_right(a, b, u64::MAX)
    } else {
        shift_right(a, b, 0)
    }
}

fn shift_right(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES], init: u64) -> [u8; INT256_BYTES] {
    let a = to_u64(a);
    let b = to_u64(b);
    let shift = (b[0] & 0xff) as u32;
    let index_offset = (shift / U64_BITS) as usize;
    let bit_offset = shift % U64_BITS;
    let mut r = [init; INT256_U64_LIMBS];
    let mut carry = if bit_offset > 0 {
        init << (U64_BITS - bit_offset)
    } else {
        0
    };
    for i in (index_offset..INT256_U64_LIMBS).rev() {
        let curr = a[i];
        r[i - index_offset] = (curr >> bit_offset) + carry;
        if bit_offset > 0 {
            carry = curr << (U64_BITS - bit_offset);
        }
    }
    from_u64(&r)
}

fn u256_slt(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let result = i256_lt(a, b);
    let mut r = [0u8; INT256_BYTES];
    r[0] = result as u8;
    r
}

fn u256_sltu(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let result = u256_lt(a, b);
    let mut r = [0u8; INT256_BYTES];
    r[0] = result as u8;
    r
}

fn u256_mul(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> [u8; INT256_BYTES] {
    let a = to_u32_arr(a);
    let b = to_u32_arr(b);
    let mut r = [0u32; INT256_WORDS];
    for i in 0..INT256_WORDS {
        let mut carry = 0u64;
        for j in 0..(INT256_WORDS - i) {
            let res = a[i] as u64 * b[j] as u64 + r[i + j] as u64 + carry;
            r[i + j] = res as u32;
            carry = res >> u32::BITS;
        }
    }
    from_u32_arr(&r)
}

// ── Comparison helpers ──────────────────────────────────────────────────────

fn u256_lt(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> bool {
    let a = to_u64(a);
    let b = to_u64(b);
    for i in (0..INT256_U64_LIMBS).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

fn i256_lt(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> bool {
    let a_sign = a[INT256_MSB] >> SIGN_BIT_SHIFT == 1;
    let b_sign = b[INT256_MSB] >> SIGN_BIT_SHIFT == 1;
    let a64 = to_u64(a);
    let b64 = to_u64(b);
    for i in (0..INT256_U64_LIMBS).rev() {
        if a64[i] != b64[i] {
            return (a64[i] < b64[i]) ^ a_sign ^ b_sign;
        }
    }
    false
}

fn u256_eq(a: &[u8; INT256_BYTES], b: &[u8; INT256_BYTES]) -> bool {
    let a = to_u64(a);
    let b = to_u64(b);
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

// ── FFI entry points ────────────────────────────────────────────────────────
//
// One specialized entry point per opcode so the C compiler emits a direct call
// for each instance (no runtime `op` switch). The per-instruction cost (1 row)
// is already counted by the per-block chip update emitted at block entry in
// the generated C, so none of these need a chip index.

/// Defines an `extern "C"` ALU entry point that reads `rs1`/`rs2`, applies
/// `$op`, and writes the result to `rd`.
macro_rules! int256_alu_fn {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// `state` must be a valid pointer to the C `RvState` struct.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u32,
            rs1_ptr: u32,
            rs2_ptr: u32,
        ) {
            let rs1 = read_int256(state, rs1_ptr);
            let rs2 = read_int256(state, rs2_ptr);
            let result = $op(&rs1, &rs2);
            write_int256(state, rd_ptr, &result);
        }
    };
}

int256_alu_fn!(rvr_ext_int256_add, u256_add);
int256_alu_fn!(rvr_ext_int256_sub, u256_sub);
int256_alu_fn!(rvr_ext_int256_xor, u256_xor);
int256_alu_fn!(rvr_ext_int256_or, u256_or);
int256_alu_fn!(rvr_ext_int256_and, u256_and);
int256_alu_fn!(rvr_ext_int256_sll, u256_sll);
int256_alu_fn!(rvr_ext_int256_srl, u256_srl);
int256_alu_fn!(rvr_ext_int256_sra, u256_sra);
int256_alu_fn!(rvr_ext_int256_slt, u256_slt);
int256_alu_fn!(rvr_ext_int256_sltu, u256_sltu);
int256_alu_fn!(rvr_ext_int256_mul, u256_mul);

/// Defines an `extern "C"` branch predicate. Returns 1 if the branch is taken.
macro_rules! int256_branch_fn {
    ($name:ident, $cond:expr) => {
        /// # Safety
        /// `state` must be a valid pointer to the C `RvState` struct.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rs1_ptr: u32, rs2_ptr: u32) -> u32 {
            let rs1 = read_int256(state, rs1_ptr);
            let rs2 = read_int256(state, rs2_ptr);
            ($cond(&rs1, &rs2)) as u32
        }
    };
}

int256_branch_fn!(rvr_ext_int256_beq, u256_eq);
int256_branch_fn!(rvr_ext_int256_bne, |a, b| !u256_eq(a, b));
int256_branch_fn!(rvr_ext_int256_blt, i256_lt);
int256_branch_fn!(rvr_ext_int256_bltu, u256_lt);
int256_branch_fn!(rvr_ext_int256_bge, |a, b| !i256_lt(a, b));
int256_branch_fn!(rvr_ext_int256_bgeu, |a, b| !u256_lt(a, b));

#[cfg(test)]
mod tests {
    use super::*;

    fn u256_from_u64s(words: [u64; INT256_U64_LIMBS]) -> [u8; INT256_BYTES] {
        from_u64(&words)
    }

    fn shift_arg(shift: u32) -> [u8; INT256_BYTES] {
        let mut bytes = [0u8; INT256_BYTES];
        bytes[..WORD_SIZE].copy_from_slice(&shift.to_le_bytes());
        bytes
    }

    #[test]
    fn test_u256_add_and_sub_across_limbs() {
        let a = u256_from_u64s([u64::MAX, u64::MAX, 0, 0]);
        let one = u256_from_u64s([1, 0, 0, 0]);
        let sum = u256_add(&a, &one);
        assert_eq!(to_u64(&sum), [0, 0, 1, 0]);

        let back = u256_sub(&sum, &one);
        assert_eq!(to_u64(&back), [u64::MAX, u64::MAX, 0, 0]);
    }

    #[test]
    fn test_shift_boundaries() {
        let value = u256_from_u64s([1, 2, 3, 4]);

        assert_eq!(to_u64(&u256_sll(&value, &shift_arg(0))), [1, 2, 3, 4]);
        assert_eq!(to_u64(&u256_sll(&value, &shift_arg(64))), [0, 1, 2, 3]);
        assert_eq!(to_u64(&u256_srl(&value, &shift_arg(64))), [2, 3, 4, 0]);

        let negative = u256_from_u64s([0, 0, 0, 1u64 << 63]);
        assert_eq!(
            to_u64(&u256_sra(&negative, &shift_arg(255))),
            [u64::MAX, u64::MAX, u64::MAX, u64::MAX]
        );
    }

    #[test]
    fn test_i256_lt_sign_boundaries() {
        let max_i256 = u256_from_u64s([u64::MAX, u64::MAX, u64::MAX, i64::MAX as u64]);
        let min_i256 = u256_from_u64s([0, 0, 0, i64::MIN as u64]);

        assert!(i256_lt(&min_i256, &max_i256));
        assert!(!i256_lt(&max_i256, &min_i256));
        assert!(!i256_lt(&max_i256, &max_i256));
    }

    #[test]
    fn test_u256_mul_wraps() {
        let max = [0xffu8; 32];
        let two = shift_arg(2);
        let product = u256_mul(&max, &two);
        assert_eq!(
            to_u64(&product),
            [u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]
        );
    }
}
