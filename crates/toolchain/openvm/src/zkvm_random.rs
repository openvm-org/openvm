use openvm_riscv_guest::{hint_buffer_bytes, hint_random, HINT_WORD_BYTES};

// The `sys_rand` sequence for one word. `sys_rand` is `std`-only, and the standard needs this
// symbol in `no_std` guests too.
#[no_mangle]
pub extern "C" fn zkvm_random_u64() -> u64 {
    let mut value = 0u64;
    hint_random(1);
    // SAFETY: `value` is writable for `HINT_WORD_BYTES` bytes.
    unsafe { hint_buffer_bytes(&mut value as *mut u64 as *mut u8, HINT_WORD_BYTES) };
    value
}
