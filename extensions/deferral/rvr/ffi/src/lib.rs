//! FFI functions for the deferral extension (CALL + OUTPUT).
//!
//! The generated C code calls only two functions: `rvr_ext_deferral_call` and
//! `rvr_ext_deferral_output`. These look up pre-computed results from
//! `OpenVmIoState.deferral` via host callbacks forwarded through `openvm_io.c`.

use std::ffi::c_void;

use rvr_openvm_ext_ffi_common::{
    ext_deferral_call_lookup, ext_deferral_output_lookup, rd_mem_words_traced, trace_chip_wrapper,
    trace_mem_access_range, wr_mem_words_traced, DEFERRAL_AS, DEFERRAL_COMMIT_NUM_BYTES,
    DEFERRAL_DIGEST_SIZE, DEFERRAL_OUTPUT_KEY_BYTES, WORD_SIZE,
};

const COMMIT_WORDS: usize = DEFERRAL_COMMIT_NUM_BYTES / WORD_SIZE;
const OUTPUT_KEY_WORDS: usize = DEFERRAL_OUTPUT_KEY_BYTES / WORD_SIZE;
const DIGEST_MEMORY_OPS: usize = DEFERRAL_DIGEST_SIZE / WORD_SIZE;

// ── FFI entry points (called by generated C code) ─────────────────────────────

/// Deferral CALL: read input_commit, look up output_key, write it.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_deferral_call(
    state: *mut c_void,
    output_ptr: u32,
    input_ptr: u32,
    def_idx: u32,
    poseidon2_chip_idx: u32,
) {
    // Read input_commit (8 words) from memory and pack into bytes.
    let mut commit_words = [0u32; COMMIT_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut commit_words);
    let mut input_commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
    for (i, &w) in commit_words.iter().enumerate() {
        input_commit[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }

    // Trace DEFERRAL_AS reads (old_input_acc + old_output_acc).
    // TODO: deferral address space elements are field elements, not u32.
    // Page tracking currently assumes u32 cells — verify that the page
    // and chunk geometry is correct for field-element-typed cells.
    let input_acc_ptr = 2 * def_idx * DEFERRAL_DIGEST_SIZE as u32;
    let output_acc_ptr = input_acc_ptr + DEFERRAL_DIGEST_SIZE as u32;
    trace_mem_access_range(state, input_acc_ptr, DIGEST_MEMORY_OPS as u32, DEFERRAL_AS);
    trace_mem_access_range(state, output_acc_ptr, DIGEST_MEMORY_OPS as u32, DEFERRAL_AS);

    // Look up pre-computed output_key via host callback.
    let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
    let found = ext_deferral_call_lookup(def_idx, input_commit.as_ptr(), output_key.as_mut_ptr());
    assert_eq!(
        found,
        1,
        "deferral CALL lookup failed: def_idx={}, input_commit={:02x?}",
        def_idx,
        &input_commit[..WORD_SIZE * 2]
    );

    // Write output_key (10 words) to memory.
    let mut key_words = [0u32; OUTPUT_KEY_WORDS];
    for (i, w) in key_words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(
            output_key[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    wr_mem_words_traced(state, output_ptr, &key_words);

    // Trace DEFERRAL_AS writes (new_input_acc + new_output_acc).
    trace_mem_access_range(state, input_acc_ptr, DIGEST_MEMORY_OPS as u32, DEFERRAL_AS);
    trace_mem_access_range(state, output_acc_ptr, DIGEST_MEMORY_OPS as u32, DEFERRAL_AS);

    trace_chip_wrapper(state, poseidon2_chip_idx, 2);
}

/// Deferral OUTPUT: read output_key, look up raw output, write it.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_deferral_output(
    state: *mut c_void,
    output_ptr: u32,
    input_ptr: u32,
    _def_idx: u32,
    output_chip_idx: u32,
    poseidon2_chip_idx: u32,
) {
    // Read output_key (10 words) from memory and unpack into bytes.
    let mut key_words = [0u32; OUTPUT_KEY_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut key_words);
    let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
    for (i, &w) in key_words.iter().enumerate() {
        output_key[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }

    let output_commit: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
        output_key[..DEFERRAL_COMMIT_NUM_BYTES].try_into().unwrap();
    let output_len =
        u64::from_le_bytes(output_key[DEFERRAL_COMMIT_NUM_BYTES..].try_into().unwrap()) as usize;

    // Look up pre-computed raw output via host callback, write directly to a local buffer.
    let mut output_raw = vec![0u8; output_len];
    let found = ext_deferral_output_lookup(
        output_commit.as_ptr(),
        output_raw.as_mut_ptr(),
        output_len as u32,
    );
    assert_eq!(
        found,
        1,
        "deferral OUTPUT lookup failed: output_commit={:02x?}",
        &output_commit[..WORD_SIZE * 2]
    );

    // Write raw output to memory in DEFERRAL_DIGEST_SIZE-byte rows. Each row
    // is independent (lookup may return contiguous bytes, but the trace API
    // is per-row), so do one batched write per row.
    let num_data_rows = output_len / DEFERRAL_DIGEST_SIZE;
    let mut row_words = [0u32; DIGEST_MEMORY_OPS];
    for row_idx in 0..num_data_rows {
        let row_byte_base = row_idx * DEFERRAL_DIGEST_SIZE;
        for (chunk_idx, w) in row_words.iter_mut().enumerate() {
            let base = row_byte_base + chunk_idx * WORD_SIZE;
            *w = u32::from_le_bytes(output_raw[base..base + WORD_SIZE].try_into().unwrap());
        }
        let row_output_ptr = output_ptr + row_byte_base as u32;
        wr_mem_words_traced(state, row_output_ptr, &row_words);
    }

    let num_rows = (output_len / DEFERRAL_DIGEST_SIZE + 1) as u32;
    if num_rows > 1 {
        trace_chip_wrapper(state, output_chip_idx, num_rows - 1);
    }
    trace_chip_wrapper(state, poseidon2_chip_idx, num_rows);
}
