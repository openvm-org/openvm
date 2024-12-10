use core::{arch::asm, cmp::Ordering, mem::MaybeUninit};

use axvm_platform::custom_insn_r;

use super::{Int256Funct7, BEQ256_FUNCT3, INT256_FUNCT3, OPCODE};

#[no_mangle]
unsafe extern "C" fn wrapping_add_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Add as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn wrapping_sub_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Sub as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn wrapping_mul_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Mul as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn bitxor_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Xor as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn bitand_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::And as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn bitor_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Or as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn wrapping_shl_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Sll as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn wrapping_shr_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Srl as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn arithmetic_shr_impl(a: *const u64, b: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Sra as u8,
        result as *mut u64,
        a as *const u64,
        b as *const u64
    );
}

#[no_mangle]
unsafe extern "C" fn eq_impl(a: *const u64, b: *const u64) -> bool {
    let mut is_equal: u32;
    asm!("li {res}, 1",
        ".insn b {opcode}, {func3}, {rs1}, {rs2}, 8",
        "li {res}, 0",
        opcode = const OPCODE,
        func3 = const BEQ256_FUNCT3,
        rs1 = in(reg) a as *const u64,
        rs2 = in(reg) b as *const u64,
        res = out(reg) is_equal
    );
    return is_equal == 1;
}

#[no_mangle]
unsafe extern "C" fn cmp_impl(a: *const u64, b: *const u64) -> Ordering {
    let mut cmp_result = MaybeUninit::<[u64; 4]>::uninit();
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Sltu as u8,
        cmp_result.as_mut_ptr(),
        a as *const u64,
        b as *const u64
    );
    let mut cmp_result = cmp_result.assume_init();
    if cmp_result[0] != 0 {
        return Ordering::Less;
    }
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Sltu as u8,
        &mut cmp_result as *mut [u64; 4],
        b as *const u64,
        a as *const u64
    );
    if cmp_result[0] != 0 {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[no_mangle]
unsafe extern "C" fn clone_impl(a: *const u64, zero: *const u64, result: *mut u64) {
    custom_insn_r!(
        OPCODE,
        INT256_FUNCT3,
        Int256Funct7::Add as u8,
        result as *mut u64,
        a as *const u64,
        zero as *const u64
    );
}
