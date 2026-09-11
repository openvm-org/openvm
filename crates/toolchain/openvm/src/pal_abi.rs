/// For rust std library compatibility, we need to define the ABI specified in
/// <https://github.com/openvm-org/rust/blob/feat/riscv64im-unknown-openvm-elf/library/std/src/sys/pal/openvm/abi.rs>
/// while we are using target = "riscv64im-unknown-openvm-elf".
use openvm_platform::{fileno::*, rust_rt::terminate};
use openvm_riscv_guest::{
    hint_buffer_bytes, hint_random, raw_print_str_from_bytes, HINT_WORD_BYTES,
};

/// Exit codes returned to the host via `terminate::<CODE>()`. The host's `sys_halt`
/// expects a `u8` user-exit byte; 0 is reserved for "successful halt from user main",
/// 1–3 mirror common abort categories, and 4 is the explicit halt sentinel used by
/// this PAL's `sys_halt` (distinct from a normal program return).
pub mod exit_code {
    pub const PANIC: u8 = 1;
    pub const UNIMP: u8 = 2;
    pub const HALT: u8 = 4;
}

// [inline(never)] is added to mitigate potentially leaking information about program execution
// through the final value of the program counter (pc) on halt where there is more than one
// location in the program where `sys_halt` is called. As long as the halt instruction only exists
// in one place within the program, the pc will always be the same invariant with input.
#[inline(never)]
#[no_mangle]
pub extern "C" fn sys_halt() -> ! {
    terminate::<{ exit_code::HALT }>()
}

/// Fill `nbytes` of `recv_buf` with random bytes from the openvm host hint stream.
///
/// # Safety
///
/// `recv_buf` must be dereferenceable for `nbytes` bytes (no alignment required).
#[no_mangle]
pub unsafe extern "C" fn sys_rand(recv_buf: *mut u8, nbytes: usize) {
    if nbytes == 0 {
        return;
    }
    hint_random(nbytes.div_ceil(HINT_WORD_BYTES));
    hint_buffer_bytes(recv_buf, nbytes);
}

/// # Safety
///
/// `msg_ptr` must be aligned and dereferenceable.
#[no_mangle]
unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) -> ! {
    raw_print_str_from_bytes(msg_ptr, len);
    terminate::<{ exit_code::PANIC }>()
}

/// # Safety
///
/// `msg_ptr` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_log(msg_ptr: *const u8, len: usize) {
    raw_print_str_from_bytes(msg_ptr, len);
}

/// Reads the given number of bytes into the given buffer, posix-style. Returns
/// the number of bytes actually read. On end of file, returns 0.
///
/// # Safety
///
/// `recv_ptr` must be dereferenceable for `nread` bytes.
#[no_mangle]
pub unsafe extern "C" fn sys_read(_fd: i32, _recv_ptr: *mut u8, _nread: usize) -> usize {
    crate::io::println("unsupported OpenVM syscall: sys_read");
    terminate::<{ exit_code::UNIMP }>()
}

/// # Safety
///
/// `write_ptr` must be dereferenceable for `nbytes` bytes.
#[no_mangle]
pub unsafe extern "C" fn sys_write(fd: i32, write_ptr: *const u8, nbytes: usize) {
    if fd == STDOUT || fd == STDERR {
        // We always print to host stdout using UTF-8 encoding.
        raw_print_str_from_bytes(write_ptr, nbytes);
    } else {
        use core::fmt::Write;
        let mut writer = crate::io::Writer;
        let _ = writeln!(writer, "sys_write to fd={fd} not supported.");
        terminate::<{ exit_code::UNIMP }>()
    }
}

/// Retrieves the value of an environment variable, and stores as much of it as
/// it can in `[recv_buf, recv_buf + recv_nbytes)`. Returns the length of the
/// value in bytes, or `usize::MAX` if the variable is not set.
///
/// This is normally called twice: once to get the length of the value, and
/// once to fill the allocated buffer.
///
/// NOTE: Repeated calls to sys_getenv are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
///
/// # Safety
///
/// `recv_buf` must be dereferenceable for `recv_nbytes` bytes; `varname` for
/// `varname_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn sys_getenv(
    _recv_buf: *mut u8,
    _recv_nbytes: usize,
    _varname: *const u8,
    _varname_len: usize,
) -> usize {
    // No env vars in the guest. Per the contract, return usize::MAX to signal "not set"
    // (returning 0 would mean "set, empty value", which std would then try to read).
    usize::MAX
}

/// Retrieves the count of arguments provided to program execution.
///
/// NOTE: Repeated calls to sys_argc are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
#[no_mangle]
pub extern "C" fn sys_argc() -> usize {
    0
}

/// Retrieves the argument with `arg_index`, and stores as much of it as it can
/// in `[out_buf, out_buf + out_nbytes)`. Returns the length of the argument in
/// bytes. If `arg_index >= argc` this syscall does not return.
///
/// This is normally called twice: once to get the length, and once to fill the
/// allocated buffer.
///
/// NOTE: Repeated calls to sys_argv are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
///
/// # Safety
///
/// `out_buf` must be dereferenceable for `out_nbytes` bytes.
#[no_mangle]
pub unsafe extern "C" fn sys_argv(
    _out_buf: *mut u8,
    _out_nbytes: usize,
    _arg_index: usize,
) -> usize {
    // sys_argc returns 0 so any call here is for an out-of-range index, which the
    // ABI documents as "does not return". Terminate explicitly.
    crate::io::println("unsupported OpenVM syscall: sys_argv");
    terminate::<{ exit_code::UNIMP }>()
}
