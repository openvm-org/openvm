//! Backend for proving and verifying mixed-matrix STARKs with univariate polynomial commitment scheme.

/// AIR builders for prover and verifier, including support for cross-matrix permutation arguments.
pub mod air_builders;
/// Types for tracking matrix in system with multiple commitments, each to multiple matrices.
pub mod commit;
/// Helper types associated to generic STARK config.
pub mod config;
pub mod interaction;
/// Proving and verifying key generation
pub mod keygen;
/// Prover implementation for partitioned multi-matrix AIRs.
pub mod prover;
pub mod rap;
pub mod utils;
pub mod verifier;

// Use jemalloc as global allocator for performance
#[cfg(all(feature = "jemalloc", unix))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Use mimalloc as global allocator
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
