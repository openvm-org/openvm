//! System exit and panic functions.

/// Exit the program with exit code 0.
pub fn exit() {
    axvm_platform::rust_rt::terminate::<0>();
}

/// Exit the program with exit code 1.
pub fn panic() {
    axvm_platform::rust_rt::terminate::<1>();
}
