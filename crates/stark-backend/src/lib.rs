//! Backend for proving and verifying mixed-matrix STARKs with univariate polynomial commitment scheme.

// Re-export all Plonky3 crates
pub use p3_air;
pub use p3_challenger;
pub use p3_commit;
pub use p3_field;
pub use p3_matrix;
pub use p3_maybe_rayon;
pub use p3_util;

/// AIR builders for prover and verifier, including support for cross-matrix permutation arguments.
pub mod air_builders;
/// Trait for stateful chip that owns trace generation
mod chip;
/// API trait for circuit prover/verifier.
pub mod circuit_api;
/// Types for tracking matrix in system with multiple commitments, each to multiple matrices.
pub mod commit;
/// Helper types associated to generic STARK config.
pub mod config;
/// Trait for STARK backend engine proving keygen, proviing, verifying API functions.
pub mod engine;
/// GKR batch prover for Grand Product and LogUp lookup arguments.
pub mod gkr;
/// Log-up permutation argument implementation as RAP.
pub mod interaction;
/// Proving and verifying key generation
pub mod keygen;
/// Polynomials
pub mod poly;
/// Prover implementation for partitioned multi-matrix AIRs.
pub mod prover;
/// Trait for RAP (Randomized AIR with Preprocessing)
pub mod rap;
/// Sum-check protocol
pub mod sumcheck;
/// Utility functions
pub mod utils;
/// Verifier implementation
pub mod verifier;

pub use chip::{Chip, ChipUsageGetter};

// Use jemalloc as global allocator for performance
#[cfg(all(feature = "jemalloc", unix, not(test)))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Use mimalloc as global allocator
#[cfg(all(feature = "mimalloc", not(test)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
