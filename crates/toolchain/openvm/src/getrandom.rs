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
