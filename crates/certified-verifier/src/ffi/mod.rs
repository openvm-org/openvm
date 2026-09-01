//! Rust side of the C adapter around the Lean VM verifier.

use std::{ffi::c_char, io, sync::Mutex};

const ERROR_CAPACITY: usize = 4096;

extern "C" {
    fn openvm_vm_verify(
        vk: *const u8,
        vk_len: usize,
        baseline: *const u8,
        baseline_len: usize,
        proof: *const u8,
        proof_len: usize,
        public_values: *const u8,
        public_values_len: usize,
        user_public_values: *const u8,
        user_public_values_len: usize,
        error_out: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

// The adapter initializes process-global Lean state lazily, and Lean's
// single-threaded reference counts are not safe across concurrent calls.
static VERIFY_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn verify(
    vk: &[u8],
    baseline: &[u8],
    proof: &[u8],
    public_values: &[u8],
    user_public_values: &[u8],
) -> io::Result<(i32, String)> {
    let _guard = VERIFY_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut error = [0u8; ERROR_CAPACITY];

    // SAFETY: All pointers remain valid for the call, their lengths match the
    // slices, the output buffer is writable, and calls are serialized above.
    let exit_code = unsafe {
        openvm_vm_verify(
            vk.as_ptr(),
            vk.len(),
            baseline.as_ptr(),
            baseline.len(),
            proof.as_ptr(),
            proof.len(),
            public_values.as_ptr(),
            public_values.len(),
            user_public_values.as_ptr(),
            user_public_values.len(),
            error.as_mut_ptr().cast(),
            error.len(),
        )
    };
    let message_len = error
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(error.len());
    let message = String::from_utf8_lossy(&error[..message_len]).into_owned();

    if exit_code < 0 {
        let detail = if message.is_empty() {
            format!("Lean FFI adapter failed with code {exit_code}")
        } else {
            message
        };
        Err(io::Error::other(detail))
    } else {
        Ok((exit_code, message))
    }
}
