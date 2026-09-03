//! Wire-format round-trip tests against the Lean decoder.
//!
//! Generates a real FibonacciAir / BabyBearPoseidon2 proof through this
//! OpenVM workspace's `openvm-stark-backend` revision, encodes
//! the five VM verifier blobs through `openvm-certified-verifier`, pipes them
//! to the Lean `vm_dump_proof` binary, and asserts parity on the structural
//! digests. Also covers tampered, empty, and wrong-version inputs.

use std::{
    io::Write,
    process::{Command, Stdio},
};

use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    proof::Proof,
    test_utils::{FibFixture, TestFixture},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, BabyBearPoseidon2RefEngine, DuplexSponge, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::{pvs::VkCommit, vk::VerificationBaseline};

use crate::{
    harness::vm_dump_proof_bin,
    magic::{
        MAGIC_PROOF, MAGIC_PUBLIC_VALUES, MAGIC_USER_PUBLIC_VALUES, MAGIC_VK, MAGIC_VM_BASELINE,
        WIRE_VERSION,
    },
    proof::write_proof,
    public_values::write_public_values,
    vk::write_vk,
    vm::{write_user_public_values_proof, write_verification_baseline},
};

const LOG_TRACE_DEGREE: usize = 5;

type SC = BabyBearPoseidon2Config;

struct Fixture {
    vk: MultiStarkVerifyingKey<SC>,
    proof: Proof<SC>,
}

/// Generate a fresh (vk, proof) pair from FibonacciAir.
fn fresh_fixture() -> Fixture {
    let params = SystemParams::new_for_testing(LOG_TRACE_DEGREE);
    let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(params);
    let n = 1usize << LOG_TRACE_DEGREE;
    let (vk, proof) = FibFixture::new(0, 1, n).keygen_and_prove(&engine);
    engine
        .verify(&vk, &proof)
        .expect("upstream FibonacciAir prove+verify must succeed");
    Fixture { vk, proof }
}

fn fixture_digest(start: u32) -> [F; DIGEST_SIZE] {
    std::array::from_fn(|i| F::from_u32(start + i as u32))
}

fn fixture_vk_commit(start: u32) -> VkCommit<F> {
    VkCommit {
        cached_commit: fixture_digest(start),
        vk_pre_hash: fixture_digest(start + DIGEST_SIZE as u32),
    }
}

fn baseline_fixture() -> VerificationBaseline {
    VerificationBaseline {
        app_exe_commit: fixture_digest(10),
        memory_dimensions: MemoryDimensions {
            addr_space_height: 3,
            address_height: 29,
        },
        num_user_pvs: 16,
        app_vk_commit: fixture_vk_commit(100),
        leaf_vk_commit: fixture_vk_commit(200),
        internal_for_leaf_vk_commit: fixture_vk_commit(300),
        internal_recursive_vk_commit: fixture_vk_commit(400),
        expected_def_hook_commit: None,
    }
}

fn user_pvs_fixture() -> UserPublicValuesProof<DIGEST_SIZE, F> {
    UserPublicValuesProof {
        proof: vec![
            fixture_digest(500),
            fixture_digest(600),
            fixture_digest(700),
        ],
        public_values: (800..816).map(F::from_u32).collect(),
        public_values_commit: fixture_digest(900),
    }
}

fn encode_vm_fixture() -> (Vec<u8>, Vec<u8>) {
    let mut baseline_bytes = Vec::new();
    write_verification_baseline(&mut baseline_bytes, &baseline_fixture())
        .expect("write_verification_baseline");
    let mut user_pvs_bytes = Vec::new();
    write_user_public_values_proof(&mut user_pvs_bytes, &user_pvs_fixture())
        .expect("write_user_public_values_proof");
    (baseline_bytes, user_pvs_bytes)
}

/// Concatenate the production five-blob input with `u32 LE` length prefixes.
fn frame_five_blobs(blobs: [&[u8]; 5]) -> Vec<u8> {
    let mut buf = Vec::new();
    for blob in blobs {
        let len = u32::try_from(blob.len()).expect("wire test blob length fits in u32");
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(blob);
    }
    buf
}

fn frame_vm_blobs(vk_bytes: &[u8], proof_bytes: &[u8], pv_bytes: &[u8]) -> Vec<u8> {
    let (baseline_bytes, user_pvs_bytes) = encode_vm_fixture();
    frame_five_blobs([
        vk_bytes,
        &baseline_bytes,
        proof_bytes,
        pv_bytes,
        &user_pvs_bytes,
    ])
}

struct DumpOutcome {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

/// Pipe `stdin_bytes` to `vm_dump_proof` and capture exit + output.
fn run_vm_dump_proof(stdin_bytes: &[u8]) -> DumpOutcome {
    let mut child = Command::new(vm_dump_proof_bin())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vm_dump_proof");
    {
        let mut child_stdin = child.stdin.take().expect("child stdin piped");
        child_stdin
            .write_all(stdin_bytes)
            .expect("write stdin to vm_dump_proof");
    }
    let output = child.wait_with_output().expect("wait vm_dump_proof");
    DumpOutcome {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Encode the fixture into three wire blobs.
fn encode_fixture(f: &Fixture) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut vk_bytes = Vec::new();
    write_vk(&mut vk_bytes, &f.vk).expect("write_vk");
    let mut proof_bytes = Vec::new();
    write_proof(&mut proof_bytes, &f.proof).expect("write_proof");
    let mut pv_bytes = Vec::new();
    write_public_values(&mut pv_bytes, &f.vk, &f.proof.public_values).expect("write_public_values");
    (vk_bytes, proof_bytes, pv_bytes)
}

/// Compute the same `vk: ...` digest line the Lean side prints.
fn vk_digest_csv(vk: &MultiStarkVerifyingKey<SC>) -> String {
    digest_csv(&vk.pre_hash)
}

fn proof_digest_csv(proof: &Proof<SC>) -> String {
    digest_csv(&proof.common_main_commit)
}

fn digest_csv(digest: &[F; 8]) -> String {
    digest
        .iter()
        .map(|v| v.as_canonical_u32().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn pv_digest_csv(public_values: &[Vec<F>]) -> String {
    public_values
        .iter()
        .map(|pv| {
            pv.iter()
                .map(|v| v.as_canonical_u32().to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn baseline_digest(baseline: &VerificationBaseline) -> String {
    let mut parts = vec![
        digest_csv(&baseline.app_exe_commit),
        baseline.memory_dimensions.addr_space_height.to_string(),
        baseline.memory_dimensions.address_height.to_string(),
        baseline.num_user_pvs.to_string(),
    ];
    for commit in [
        &baseline.app_vk_commit,
        &baseline.leaf_vk_commit,
        &baseline.internal_for_leaf_vk_commit,
        &baseline.internal_recursive_vk_commit,
    ] {
        parts.push(digest_csv(&commit.cached_commit));
        parts.push(digest_csv(&commit.vk_pre_hash));
    }
    parts.join("|")
}

fn user_pvs_digest(proof: &UserPublicValuesProof<DIGEST_SIZE, F>) -> String {
    let authentication_path = proof
        .proof
        .iter()
        .map(digest_csv)
        .collect::<Vec<_>>()
        .join(";");
    let public_values = proof
        .public_values
        .iter()
        .map(|value| value.as_canonical_u32().to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{authentication_path}|{public_values}|{}",
        digest_csv(&proof.public_values_commit)
    )
}

/// Locate the first byte of the body (right after the 8-byte header)
/// in a blob produced by the proof-wire encoder.
const HEADER_LEN: usize = 8;

// =====================================================================
// Green path
// =====================================================================

#[test]
fn green_proof() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 0,
        "vm_dump_proof must accept a freshly encoded fixture; \
         stdout={:?} stderr={:?}",
        outcome.stdout, outcome.stderr
    );
    let expected = format!("proof: {}", proof_digest_csv(&fixture.proof));
    let found = outcome.stdout.lines().any(|line| line == expected.as_str());
    assert!(
        found,
        "expected stdout to contain {expected:?}; stdout was:\n{}",
        outcome.stdout
    );
}

#[test]
fn green_vk() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(outcome.exit_code, 0, "vm_dump_proof rejected fixture");
    let expected = format!("vk: {}", vk_digest_csv(&fixture.vk));
    let found = outcome.stdout.lines().any(|line| line == expected.as_str());
    assert!(
        found,
        "expected stdout to contain {expected:?}; stdout was:\n{}",
        outcome.stdout
    );
}

#[test]
fn green_public_values() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(outcome.exit_code, 0, "vm_dump_proof rejected fixture");
    let expected = format!("pv: {}", pv_digest_csv(&fixture.proof.public_values));
    let found = outcome.stdout.lines().any(|line| line == expected.as_str());
    assert!(
        found,
        "expected stdout to contain {expected:?}; stdout was:\n{}",
        outcome.stdout
    );
}

#[test]
fn green_vm_baseline() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let baseline = baseline_fixture();
    let mut baseline_bytes = Vec::new();
    write_verification_baseline(&mut baseline_bytes, &baseline).expect("write VM baseline");
    let mut user_pvs_bytes = Vec::new();
    write_user_public_values_proof(&mut user_pvs_bytes, &user_pvs_fixture())
        .expect("write user-PV proof");

    let outcome = run_vm_dump_proof(&frame_five_blobs([
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    ]));
    assert_eq!(outcome.exit_code, 0, "vm_dump_proof rejected fixture");
    let expected = format!("baseline: {}", baseline_digest(&baseline));
    assert!(
        outcome.stdout.lines().any(|line| line == expected),
        "expected stdout to contain {expected:?}; stdout was:\n{}",
        outcome.stdout
    );
}

#[test]
fn green_user_pvs_proof() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let user_pvs = user_pvs_fixture();
    let mut baseline_bytes = Vec::new();
    write_verification_baseline(&mut baseline_bytes, &baseline_fixture())
        .expect("write VM baseline");
    let mut user_pvs_bytes = Vec::new();
    write_user_public_values_proof(&mut user_pvs_bytes, &user_pvs).expect("write user-PV proof");

    let outcome = run_vm_dump_proof(&frame_five_blobs([
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    ]));
    assert_eq!(outcome.exit_code, 0, "vm_dump_proof rejected fixture");
    let expected = format!("user-pvs: {}", user_pvs_digest(&user_pvs));
    assert!(
        outcome.stdout.lines().any(|line| line == expected),
        "expected stdout to contain {expected:?}; stdout was:\n{}",
        outcome.stdout
    );
}

// =====================================================================
// Tampered body (single byte flip) -> decodeFailure (exit 13)
// =====================================================================

#[test]
fn tampered_proof() {
    let fixture = fresh_fixture();
    let (vk_bytes, mut proof_bytes, pv_bytes) = encode_fixture(&fixture);
    // Flip the highest bit of the first FBB byte in the proof body
    // (right after the 8-byte header). For a canonical BabyBear `u32 <
    // p = 2013265921 < 2^31`, flipping the highest byte's high bit will
    // push the value to >= 2^31, which is greater than `p`, so the
    // Lean decoder must reject it as decodeFailure
    // "non-canonical babybear value".
    let target = HEADER_LEN + 3; // 4th byte of the first u32 (LE high byte)
    proof_bytes[target] ^= 0x80;
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 13,
        "expected decodeFailure (exit 13) from tampered proof, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
    assert!(
        outcome.stderr.contains("proof parse error"),
        "expected stderr mention of proof; got {:?}",
        outcome.stderr
    );
}

#[test]
fn tampered_vk() {
    let fixture = fresh_fixture();
    let (mut vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    // `preHash` is the very last field of the vk blob: 8 × FBB (32 bytes
    // total). The final byte of `vk_bytes` is therefore the high byte
    // of the last FBB in `preHash`. Canonical BabyBear satisfies
    // `u32 < p = 2013265921 < 2^31`, so canonical high bytes are < 0x78.
    // XOR-ing the high bit forces the u32 above `2^31 > p`, so the
    // decoder rejects it as `decodeFailure "non-canonical babybear
    // value"`. Earlier vk fields (SystemParams' lSkip / nStack / ...)
    // are plain Nats with no bound checks in the spec — tampering them
    // would round-trip cleanly, which is why we target preHash here.
    let target = vk_bytes.len() - 1;
    vk_bytes[target] ^= 0x80;
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 13,
        "expected decodeFailure (exit 13) from tampered vk, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
    assert!(
        outcome.stderr.contains("vk parse error"),
        "expected stderr mention of vk; got {:?}",
        outcome.stderr
    );
}

#[test]
fn tampered_public_values() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, mut pv_bytes) = encode_fixture(&fixture);
    // The first byte of the body is the airCount u32 (LE). Flipping it
    // changes the airCount, which the decoder cross-checks against
    // `vk.airCount` and rejects with `pv-air-count`.
    let target = HEADER_LEN;
    pv_bytes[target] ^= 0xFF;
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 13,
        "expected decodeFailure (13) from tampered pv, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
    assert!(
        outcome.stderr.contains("pv-air-count") || outcome.stderr.contains("pv-air-len"),
        "expected pv-air-count or pv-air-len in stderr; got {:?}",
        outcome.stderr
    );
}

#[test]
fn tampered_vm_baseline() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let (mut baseline_bytes, user_pvs_bytes) = encode_vm_fixture();
    baseline_bytes[HEADER_LEN + 3] ^= 0x80;
    let outcome = run_vm_dump_proof(&frame_five_blobs([
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    ]));
    assert_eq!(outcome.exit_code, 13);
    assert!(outcome.stderr.contains("baseline parse error"));
}

#[test]
fn tampered_user_pvs_proof() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let (baseline_bytes, mut user_pvs_bytes) = encode_vm_fixture();
    let last = user_pvs_bytes.last_mut().expect("nonempty user-PV blob");
    *last ^= 0x80;
    let outcome = run_vm_dump_proof(&frame_five_blobs([
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    ]));
    assert_eq!(outcome.exit_code, 13);
    assert!(outcome.stderr.contains("user-PV proof parse error"));
}

#[test]
fn noncanonical_public_value_row_lengths_are_rejected() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let declared_arity = fixture.vk.inner.per_air[0].params.num_public_values;
    assert!(
        declared_arity > 1,
        "fixture must expose an intermediate row length"
    );
    let row_length_offset = HEADER_LEN + 4; // Skip the public-values AIR count.

    for invalid_length in [1usize, declared_arity + 1] {
        let mut malformed = pv_bytes.clone();
        malformed[row_length_offset..row_length_offset + 4]
            .copy_from_slice(&(invalid_length as u32).to_le_bytes());
        let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &malformed));
        assert_eq!(
            outcome.exit_code, 13,
            "row length {invalid_length} should fail decoding; stderr={:?}",
            outcome.stderr
        );
        assert!(
            outcome.stderr.contains("pv-air-len"),
            "row length {invalid_length} should report pv-air-len; stderr={:?}",
            outcome.stderr
        );
    }
}

// =====================================================================
// Empty body -> unexpectedEnd (exit 12) for proof and vk; pv lands
// on `pv-air-count` mismatch (or unexpectedEnd if the body is too
// short to even contain the airCount field).
// =====================================================================

#[test]
fn empty_proof_buffer() {
    let fixture = fresh_fixture();
    let (vk_bytes, _proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &[], &pv_bytes));
    assert!(
        outcome.exit_code == 10 || outcome.exit_code == 12,
        "empty proof must surface magicMismatch (10) or unexpectedEnd (12), got {}; stderr={:?}",
        outcome.exit_code,
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("proof parse error"),
        "expected stderr mention of proof; got {:?}",
        outcome.stderr
    );
}

#[test]
fn empty_vk_buffer() {
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&[], &[], &[]));
    assert!(
        outcome.exit_code == 10 || outcome.exit_code == 12,
        "empty vk must surface magicMismatch (10) or unexpectedEnd (12), got {}; stderr={:?}",
        outcome.exit_code,
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("vk parse error"),
        "expected stderr mention of vk; got {:?}",
        outcome.stderr
    );
}

#[test]
fn empty_public_values_buffer() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, _pv_bytes) = encode_fixture(&fixture);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &[]));
    assert!(
        outcome.exit_code == 10 || outcome.exit_code == 12,
        "empty pv must surface magicMismatch (10) or unexpectedEnd (12), got {}; stderr={:?}",
        outcome.exit_code,
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("public-values parse error"),
        "expected stderr mention of pv; got {:?}",
        outcome.stderr
    );
}

// =====================================================================
// Wrong schema version -> versionMismatch (exit 11)
// =====================================================================

fn write_version_word(buf: &mut [u8], version: u32) {
    // The version word follows the 4-byte magic; bytes 4..8 are LE u32.
    buf[4..8].copy_from_slice(&version.to_le_bytes());
}

#[test]
fn wrong_proof_version() {
    let fixture = fresh_fixture();
    let (vk_bytes, mut proof_bytes, pv_bytes) = encode_fixture(&fixture);
    write_version_word(&mut proof_bytes, 0xDEAD_BEEF);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 11,
        "expected versionMismatch (11) from bad proof version, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
    assert!(
        outcome.stderr.contains("versionMismatch"),
        "expected versionMismatch in stderr; got {:?}",
        outcome.stderr
    );
}

#[test]
fn wrong_vk_version() {
    let fixture = fresh_fixture();
    let (mut vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    write_version_word(&mut vk_bytes, 0xDEAD_BEEF);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 11,
        "expected versionMismatch (11) from bad vk version, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
}

#[test]
fn wrong_public_values_version() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, mut pv_bytes) = encode_fixture(&fixture);
    write_version_word(&mut pv_bytes, 0xDEAD_BEEF);
    let outcome = run_vm_dump_proof(&frame_vm_blobs(&vk_bytes, &proof_bytes, &pv_bytes));
    assert_eq!(
        outcome.exit_code, 11,
        "expected versionMismatch (11) from bad pv version, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
}

// =====================================================================
// Magic header sanity (catches accidental swap)
// =====================================================================

#[test]
fn magic_bytes_match_proof_blob_prefix() {
    let fixture = fresh_fixture();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let (baseline_bytes, user_pvs_bytes) = encode_vm_fixture();
    assert_eq!(&vk_bytes[..4], &MAGIC_VK);
    assert_eq!(&baseline_bytes[..4], &MAGIC_VM_BASELINE);
    assert_eq!(&proof_bytes[..4], &MAGIC_PROOF);
    assert_eq!(&pv_bytes[..4], &MAGIC_PUBLIC_VALUES);
    assert_eq!(&user_pvs_bytes[..4], &MAGIC_USER_PUBLIC_VALUES);
    for blob in [
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    ] {
        assert_eq!(
            u32::from_le_bytes(blob[4..8].try_into().unwrap()),
            WIRE_VERSION,
            "expected wire version word"
        );
    }
}
