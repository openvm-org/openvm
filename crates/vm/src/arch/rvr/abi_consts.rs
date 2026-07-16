//! Compatibility guards for constants owned by external configurations.

#[cfg(any(feature = "test-utils", feature = "cuda"))]
use openvm_instructions::VM_DIGEST_WIDTH;

#[cfg(any(feature = "test-utils", feature = "cuda"))]
const _: () =
    assert!(VM_DIGEST_WIDTH == openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE,);
