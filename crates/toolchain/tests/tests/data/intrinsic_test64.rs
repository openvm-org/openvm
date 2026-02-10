#![no_std]
#![no_main]
use core::panic::PanicInfo;

// OpenVM custom opcode definitions
// SYSTEM_OPCODE = custom-0 = 0x0b
const SYSTEM_OPCODE: u8 = 0x0b;
const TERMINATE_FUNCT3: u8 = 0b000;
const HINT_FUNCT3: u8 = 0b001;
const PHANTOM_FUNCT3: u8 = 0b011;

// Phantom sub-opcodes (imm field)
const PHANTOM_HINT_INPUT: u16 = 0;  // PhantomImm::HintInput
const PHANTOM_PRINT_STR: u16 = 1;   // PhantomImm::PrintStr

// HINT_STORED imm
const HINT_STORED_IMM: u16 = 0;
// HINT_BUFFER imm
const HINT_BUFFER_IMM: u16 = 1;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    terminate::<1>();
    loop {}
}

/// Emit TERMINATE with exit code (must be a const generic)
#[inline(always)]
fn terminate<const EXIT_CODE: u8>() {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, x0, x0, {ec}",
            opcode = const SYSTEM_OPCODE,
            funct3 = const TERMINATE_FUNCT3,
            ec = const EXIT_CODE,
        );
    }
}

/// Emit PHANTOM(HintInput) — tell the host to prepare hint input
#[inline(always)]
fn phantom_hint_input() {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, x0, x0, {imm}",
            opcode = const SYSTEM_OPCODE,
            funct3 = const PHANTOM_FUNCT3,
            imm = const PHANTOM_HINT_INPUT,
        );
    }
}

/// Emit PHANTOM(PrintStr) — tell the host to print a string at (ptr, len)
#[inline(always)]
fn phantom_print_str(ptr: *const u8, len: usize) {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, {rs1}, {imm}",
            opcode = const SYSTEM_OPCODE,
            funct3 = const PHANTOM_FUNCT3,
            rd = in(reg) ptr,
            rs1 = in(reg) len,
            imm = const PHANTOM_PRINT_STR,
        );
    }
}

/// Emit HINT_STORED — store one dword from hint stream to memory at *rd
#[inline(always)]
fn hint_stored(dst: *mut u64) {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, x0, {imm}",
            opcode = const SYSTEM_OPCODE,
            funct3 = const HINT_FUNCT3,
            rd = in(reg) dst,
            imm = const HINT_STORED_IMM,
        );
    }
}

/// Emit HINT_BUFFER — store `len` dwords from hint stream to memory at *ptr
#[inline(always)]
fn hint_buffer(ptr: *mut u8, len: usize) {
    unsafe {
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, {rs1}, {imm}",
            opcode = const SYSTEM_OPCODE,
            funct3 = const HINT_FUNCT3,
            rd = in(reg) ptr,
            rs1 = in(reg) len,
            imm = const HINT_BUFFER_IMM,
        );
    }
}

fn main() {
    // Test 1: PHANTOM(PrintStr) — emit the opcode for printing
    let msg = b"hello from openvm\n";
    phantom_print_str(msg.as_ptr(), msg.len());

    // Test 2: PHANTOM(HintInput) — request hint data from host
    phantom_hint_input();

    // Test 3: HINT_STORED — store a single dword from hint stream
    let mut val: u64 = 0;
    hint_stored(&mut val as *mut u64);

    // Test 4: HINT_BUFFER — store multiple dwords from hint stream
    let mut buf = [0u8; 64];
    hint_buffer(buf.as_mut_ptr(), 8); // 8 dwords = 64 bytes

    // Test 5: Some computation using the hint data, then terminate
    // Do a trivial operation so the compiler doesn't optimize away the reads
    let sum = val.wrapping_add(buf[0] as u64);
    if sum == 0 {
        // branch just to keep the compiler from eliminating dead code
        let msg2 = b"sum is zero\n";
        phantom_print_str(msg2.as_ptr(), msg2.len());
    }

    // Test 6: TERMINATE with exit code 0 (success)
    terminate::<0>();
}

// Entry point — minimal setup, call main
mod io;
