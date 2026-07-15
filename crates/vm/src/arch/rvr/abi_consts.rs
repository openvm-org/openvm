//! Compatibility guards for constants owned by external configurations.

#[cfg(any(feature = "test-utils", feature = "cuda"))]
use openvm_instructions::DIGEST_WIDTH;

// ── External Poseidon2 digest size ─────────────────────────────────────
// Gated: plain `rvr` doesn't pull `openvm-stark-sdk`.
#[cfg(any(feature = "test-utils", feature = "cuda"))]
const _: () = assert!(DIGEST_WIDTH == openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE,);
