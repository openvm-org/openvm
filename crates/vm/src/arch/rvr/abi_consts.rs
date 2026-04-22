//! Drift guards for constants redefined in the rvr crates (which cannot
//! import from openvm-circuit without creating a cycle).
//!
//! TODO: decide whether any redefinition can be replaced with a direct
//! import — e.g. by moving the canonical constant into a leaf crate.
//!
//! TODO(defaults): `DEFAULT_PAGE_BITS` / `DEFAULT_SEGMENT_CHECK_INSNS` are
//! tunable defaults, not invariants — decide whether to keep them as
//! `const`s or restore runtime plumbing for just these two.

use openvm_instructions::{
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    DEFERRAL_AS,
};
use rvr_openvm_ext_ffi_common as ffi;

use crate::{
    arch::execution_mode::metered::{
        ctx::DEFAULT_PAGE_BITS, segment_ctx::DEFAULT_SEGMENT_CHECK_INSNS,
    },
    system::memory::{merkle::public_values::PUBLIC_VALUES_AS, CHUNK},
};

// ── rvr-openvm-ext-ffi-common address-space identifiers ────────────────
const _: () = assert!(ffi::AS_REGISTER == RV32_REGISTER_AS);
const _: () = assert!(ffi::AS_MEMORY == RV32_MEMORY_AS);
const _: () = assert!(ffi::AS_PUBLIC_VALUES == PUBLIC_VALUES_AS);
const _: () = assert!(ffi::DEFERRAL_AS == DEFERRAL_AS);

// ── rvr-openvm-ext-ffi-common word / digest sizes ──────────────────────
const _: () = assert!(ffi::WORD_SIZE == openvm_platform::WORD_SIZE);
// Gated: plain `rvr` doesn't pull `openvm-stark-sdk`.
#[cfg(any(feature = "test-utils", feature = "cuda"))]
const _: () = assert!(
    ffi::DEFERRAL_DIGEST_SIZE == openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE,
);

// ── rvr-openvm-ext-ffi-common memory / metered-execution layout ────────
const _: () = assert!(ffi::MEM_BITS == openvm_platform::memory::MEM_BITS);
const _: () = assert!(ffi::CHUNK == CHUNK);
const _: () = assert!(ffi::DEFAULT_PAGE_BITS == DEFAULT_PAGE_BITS);
const _: () = assert!(ffi::DEFAULT_SEGMENT_CHECK_INSNS as u64 == DEFAULT_SEGMENT_CHECK_INSNS);
