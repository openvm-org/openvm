//! [getrandom] custom backend implementations. The implementations are feature-gated. The default
//! feature enables "getrandom-unsupported", which is a backend that always errors. This should be
//! used when `getrandom` is never called but pulled in as a dependency unavoidably. If no feature
//! is enabled, then no custom implementation is registered, and the user must supply their own as
//! described in the [getrandom] documentation.

#[cfg(feature = "getrandom-unsupported")]
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

/// This entrypoint for getrandom is used for versions < 0.3
// The ABI is defined here: https://github.com/rust-random/getrandom/blob/ce4144b2c16fe1422037c93e267e6a52336e0834/src/custom.rs#L74
// @dev If you try to use the `getrandom_v02::Error`, it somehow triggers std library
#[cfg(feature = "getrandom-unsupported")]
#[no_mangle]
unsafe fn __getrandom_custom(dest: *mut u8, len: usize) -> u32 {
    __getrandom_v03_custom(dest, len)
        .map_err(|e| e.raw_os_error().unwrap_or(2))
        .err()
        .unwrap_or(0) as u32
}
