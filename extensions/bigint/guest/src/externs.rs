use core::{arch::asm, cmp::Ordering, mem::MaybeUninit};

use openvm_platform::custom_insn_r;

use super::{Int256Funct7, BEQ256_FUNCT3, INT256_FUNCT3, OPCODE};

#[no_mangle]
unsafe extern "C" fn zkvm_u256_wrapping_add_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Add as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_wrapping_sub_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Sub as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_wrapping_mul_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Mul as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_bitxor_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Xor as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_bitand_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::And as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_bitor_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Or as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_wrapping_shl_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Sll as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_wrapping_shr_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Srl as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_arithmetic_shr_impl(result: *mut u8, a: *const u8, b: *const u8) {
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Sra as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_eq_impl(a: *const u8, b: *const u8) -> bool {
    let mut is_equal: u32;
    asm!("li {res}, 1",
        ".insn b {opcode}, {func3}, {rs1}, {rs2}, 8",
        "li {res}, 0",
        opcode = const OPCODE,
        func3 = const BEQ256_FUNCT3,
        rs1 = in(reg) a as *const u8,
        rs2 = in(reg) b as *const u8,
        res = out(reg) is_equal
    );
    return is_equal == 1;
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_cmp_impl(a: *const u8, b: *const u8) -> Ordering {
    let mut cmp_result = MaybeUninit::<[u8; 32]>::uninit();
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Sltu as u8,
        rd = In cmp_result.as_mut_ptr(),
        rs1 = In a as *const u8,
        rs2 = In b as *const u8
    );
    let mut cmp_result = cmp_result.assume_init();
    if cmp_result[0] != 0 {
        return Ordering::Less;
    }
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7 = Int256Funct7::Sltu as u8,
        rd = In &mut cmp_result as *mut _,
        rs1 = In b as *const u8,
        rs2 = In a as *const u8
    );
    if cmp_result[0] != 0 {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[no_mangle]
unsafe extern "C" fn zkvm_u256_clone_impl(result: *mut u8, a: *const u8) {
    let zero = &[0u8; 32] as *const _ as *const u8;
    custom_insn_r!(
        opcode = OPCODE,
        funct3 = INT256_FUNCT3,
        funct7= Int256Funct7::Add as u8,
        rd = In result as *mut u8,
        rs1 = In a as *const u8,
        rs2 = In zero
    );
}
