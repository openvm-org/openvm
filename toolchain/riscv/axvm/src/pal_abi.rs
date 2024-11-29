/// For rust std library compatibility, we need to define the ABI specified in
/// <https://github.com/rust-lang/rust/blob/3dc1b9f5c00ca5535505c1ec46ccd43b8d9cfa19/library/std/src/sys/pal/zkvm/abi.rs>
/// while we are using target = "riscv32im-risc0-zkvm-elf".
/// This will be removed once a dedicated rust toolchain is used because axVM does not handle system
/// operations in the same way: there is no operating system and even the standard library should be
/// directly handled with intrinsics.
use axvm_platform::{
    constants::{Custom0Funct3, PhantomImm, CUSTOM_0},
    memory::sys_alloc_aligned,
    rust_rt::terminate,
    WORD_SIZE,
};

use crate::intrinsics::raw_print_str_from_bytes;

pub const DIGEST_WORDS: usize = 8;
// Limit syscall buffers so that the Executor doesn't get into an infinite
// split situation.
pub const MAX_BUF_BYTES: usize = 4 * 1024;
pub const MAX_BUF_WORDS: usize = MAX_BUF_BYTES / WORD_SIZE;

mod exit_code {
    pub const SUCCESS: u8 = 0;
    pub const PANIC: u8 = 1;
    // Temporarily use 4 to detect if halt is called.
    pub const HALT: u8 = 4;
    pub const PAUSE: u8 = 5;
}
/// # Safety
///
/// `out_state` must be aligned and dereferenceable.
// [inline(never)] is added to mitigate potentially leaking information about program execution
// through the final value of the program counter (pc) on halt where there is more than one
// location in the program where `sys_halt` is called. As long as the halt instruction only exists
// in one place within the program, the pc will always be the same invariant with input.
#[inline(never)]
#[no_mangle]
pub extern "C" fn sys_halt(_user_exit: u8, _out_state: *const [u32; DIGEST_WORDS]) -> ! {
    terminate::<{ exit_code::HALT }>();
    unreachable!()
}

#[no_mangle]
pub extern "C" fn sys_output(_output_id: u32, _output_value: u32) {
    unimplemented!()
}

/// # Safety
///
/// `out_state`, `in_state`, `block1_ptr`, and `block2_ptr` must be aligned and
/// dereferenceable.
#[inline(always)]
#[no_mangle]
pub unsafe extern "C" fn sys_sha_compress(
    _out_state: *mut [u32; DIGEST_WORDS],
    _in_state: *const [u32; DIGEST_WORDS],
    _block1_ptr: *const [u32; DIGEST_WORDS],
    _block2_ptr: *const [u32; DIGEST_WORDS],
) {
    unreachable!("sha_compress should not be part of PAL")
}

/// # Safety
///
/// `out_state`, `in_state`, and `buf` must be aligned and dereferenceable.
#[inline(always)]
#[no_mangle]
pub unsafe extern "C" fn sys_sha_buffer(
    _out_state: *mut [u32; DIGEST_WORDS],
    _in_state: *const [u32; DIGEST_WORDS],
    _buf: *const u8,
    _count: u32,
) {
    unreachable!("sha_buffer should not be part of PAL")
}

/// # Safety
///
/// `recv_buf` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_rand(_recv_buf: *mut u32, _words: usize) {
    unimplemented!()
}

/// # Safety
///
/// `msg_ptr` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) -> ! {
    raw_print_str_from_bytes(msg_ptr, len);
    terminate::<{ exit_code::PANIC }>();
    unreachable!()
}

/// # Safety
///
/// `msg_ptr` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_log(msg_ptr: *const u8, len: usize) {
    raw_print_str_from_bytes(msg_ptr, len);
}

#[no_mangle]
pub extern "C" fn sys_cycle_count() -> u64 {
    todo!()
}

/// Reads the given number of bytes into the given buffer, posix-style.  Returns
/// the number of bytes actually read.  On end of file, returns 0.
///
/// Like POSIX read, this is not guaranteed to read all bytes
/// requested.  If we haven't reached EOF, it is however guaranteed to
/// read at least one byte.
///
/// Users should prefer a higher-level abstraction.
///
/// # Safety
///
/// `recv_ptr` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_read(_fd: u32, _recv_ptr: *mut u8, _nread: usize) -> usize {
    todo!()
}

/// Reads up to the given number of words into the buffer [recv_buf,
/// recv_buf + nwords).  Returns the number of bytes actually read.
/// sys_read_words is a more efficient interface than sys_read, but
/// varies from POSIX semantics.  Notably:
///
/// * The read length is specified in words, not bytes.  (The output
/// length is still returned in bytes)
///
/// * If not all data is available, `sys_read_words` will return a short read.
///
/// * recv_buf must be word-aligned.
///
/// * Return a short read in the case of EOF mid-way through.
///
/// # Safety
///
/// `recv_ptr' must be a word-aligned pointer and point to a region of
/// `nwords' size.
#[no_mangle]
pub unsafe extern "C" fn sys_read_words(_fd: u32, _recv_ptr: *mut u32, _nwords: usize) -> usize {
    todo!()
}

/// # Safety
///
/// `write_ptr` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_write(_fd: u32, _write_ptr: *const u8, _nbytes: usize) {
    todo!()
}

/// Retrieves the value of an environment variable, and stores as much
/// of it as it can it in the memory at [out_words, out_words +
/// out_nwords).
///
/// Returns the length of the value, in bytes, or usize::MAX if the variable is
/// not set.
///
/// This is normally called twice to read an environment variable:
/// Once to get the length of the value, and once to fill in allocated
/// memory.
///
/// NOTE: Repeated calls to sys_getenv are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
///
/// # Safety
///
/// `out_words` and `varname` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_getenv(
    _out_words: *mut u32,
    _out_nwords: usize,
    _varname: *const u8,
    _varname_len: usize,
) -> usize {
    todo!()
}

/// Retrieves the count of arguments provided to program execution.
///
/// NOTE: Repeated calls to sys_argc are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
#[no_mangle]
pub extern "C" fn sys_argc() -> usize {
    todo!()
}

/// Retrieves the argument with arg_index, and stores as much
/// of it as it can it in the memory at [out_words, out_words +
/// out_nwords).
///
/// Returns the length, in bytes, of the argument string. If the requested
/// argument index does not exist (i.e. `arg_index` >= argc) then this syscall
/// will not return.
///
/// This is normally called twice to read an argument: Once to get the length of
/// the value, and once to fill in allocated memory.
///
/// NOTE: Repeated calls to sys_argv are not guaranteed to result in the same
/// data being returned. Returned data is entirely in the control of the host.
///
/// # Safety
///
/// `out_words` must be aligned and dereferenceable.
#[no_mangle]
pub unsafe extern "C" fn sys_argv(
    _out_words: *mut u32,
    _out_nwords: usize,
    _arg_index: usize,
) -> usize {
    todo!()
}

#[no_mangle]
#[deprecated]
pub extern "C" fn sys_alloc_words(nwords: usize) -> *mut u32 {
    unsafe { sys_alloc_aligned(WORD_SIZE * nwords, WORD_SIZE) as *mut u32 }
}

// sys_alloc_aligned is already extern no_mangle exported from axvm_platform::memory
