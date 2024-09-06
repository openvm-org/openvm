//! Backend for proving and verifying mixed-matrix STARKs with univariate polynomial commitment scheme.

// Re-export all Plonky3 crates
pub use p3_air;
pub use p3_challenger;
pub use p3_commit;
pub use p3_field;
pub use p3_matrix;
pub use p3_maybe_rayon;
pub use p3_uni_stark;
pub use p3_util;

/// AIR builders for prover and verifier, including support for cross-matrix permutation arguments.
pub mod air_builders;
/// Types for tracking matrix in system with multiple commitments, each to multiple matrices.
pub mod commit;
/// Helper types associated to generic STARK config.
pub mod config;
/// Trait for STARK backend engine proving keygen, proviing, verifying API functions.
pub mod engine;
/// Log-up permutation argument implementation as RAP.
pub mod interaction;
/// Proving and verifying key generation
pub mod keygen;
/// Prover implementation for partitioned multi-matrix AIRs.
pub mod prover;
/// Trait for RAP (Randomized AIR with Preprocessing)
pub mod rap;
/// Utility functions
pub mod utils;
/// Verifier implementation
pub mod verifier;

// Use jemalloc as global allocator for performance
#[cfg(all(feature = "jemalloc", unix))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Use mimalloc as global allocator
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
