#[cfg(not(feature = "heap-embedded-alloc"))]
mod bump;

#[cfg(feature = "heap-embedded-alloc")]
pub mod embedded;
