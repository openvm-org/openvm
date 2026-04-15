//! Rust-side handle for the keccak-f[1600] asm staticlib (built in `build.rs`).

#[cfg(all(feature = "backend-xkcp", feature = "backend-openssl"))]
compile_error!("pick exactly one keccak backend (backend-openssl / backend-xkcp)");

unsafe extern "C" {
    /// Apply keccak-f[1600] to a 25-lane u64 state in place.
    pub fn rvr_keccak_f1600(state: *mut u64);
}

/// Name of the backend compiled into this crate.
pub const ACTIVE_BACKEND: &str = {
    #[cfg(feature = "backend-openssl")]
    {
        "openssl"
    }
    #[cfg(feature = "backend-xkcp")]
    {
        "xkcp"
    }
};
