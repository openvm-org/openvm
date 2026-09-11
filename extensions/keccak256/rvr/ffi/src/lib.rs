//! Rust-side handle for the keccak-f\[1600\] asm staticlib (built in `build.rs`).

unsafe extern "C" {
    /// Apply keccak-f\[1600\] to a 25-lane u64 state in place.
    pub fn rvr_keccak_f1600(state: *mut u64);
}
