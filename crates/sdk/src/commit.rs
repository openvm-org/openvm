use std::{array::from_fn, sync::Arc};

use num_bigint::BigUint;
use openvm_circuit::arch::{instructions::exe::VmExe, MemoryConfig};
pub use openvm_circuit::system::program::trace::VmCommittedExe;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::p3_field::FieldAlgebra,
    p3_baby_bear::BabyBear,
    p3_bn254_fr::Bn254Fr,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use tracing::instrument;

use crate::{types::BN254_BYTES, F, SC};

/// Wrapper for an array of big-endian bytes, representing an unsigned big integer. Each commit can
/// be converted to a Bn254Fr using the trivial identification as natural numbers or into a `u32`
/// digest by decomposing the big integer base-`F::MODULUS`.
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommitBytes(#[serde_as(as = "serde_with::hex::Hex")] [u8; BN254_BYTES]);

impl CommitBytes {
    pub fn new(bytes: [u8; BN254_BYTES]) -> Self {
        Self(bytes)
    }

    pub fn as_slice(&self) -> &[u8; BN254_BYTES] {
        &self.0
    }

    pub fn to_bn254(&self) -> Bn254Fr {
        bytes_to_bn254(&self.0)
    }

    pub fn to_u32_digest(&self) -> [u32; DIGEST_SIZE] {
        bytes_to_u32_digest(&self.0)
    }

    pub fn from_bn254(bn254: Bn254Fr) -> Self {
        Self(bn254_to_bytes(bn254))
    }

    pub fn from_u32_digest(digest: &[u32; DIGEST_SIZE]) -> Self {
        Self(u32_digest_to_bytes(digest))
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }
}

impl std::fmt::Display for CommitBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

/// `AppExecutionCommit` has all the commitments users should check against the final proof.
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppExecutionCommit {
    /// Commitment of the executable. In base-F::MODULUS, it's computed as
    /// compress(
    ///     compress(
    ///         hash(app_program_commit),
    ///         hash(init_memory_commit)
    ///     ),
    ///     hash(right_pad(pc_start, 0))
    /// )
    /// `right_pad` example, if pc_start = 123, right_pad(pc_start, 0) = \[123,0,0,0,0,0,0,0\]
    pub app_exe_commit: CommitBytes,

    /// Commitment of the leaf VM verifier program which commits the VmConfig of App VM.
    // Internal verifier will verify `app_vm_commit`.
    // Internally this is also known as `leaf_verifier_program_commit`.
    pub app_vm_commit: CommitBytes,
}

impl AppExecutionCommit {
    /// Users should use this function to compute `AppExecutionCommit` and check it against the
    /// final proof.
    #[instrument(name = "AppExecutionCommit::compute", skip_all)]
    pub fn compute<SC: StarkGenericConfig>(
        app_memory_config: &MemoryConfig,
        app_exe: &VmExe<Val<SC>>,
        app_program_commit: Com<SC>,
        leaf_verifier_program_commit: Com<SC>,
    ) -> Self
    where
        Com<SC>: AsRef<[Val<SC>; DIGEST_SIZE]>
            + From<[Val<SC>; DIGEST_SIZE]>
            + Into<[Val<SC>; DIGEST_SIZE]>,
        Val<SC>: PrimeField32,
    {
        let exe_commit: [Val<SC>; DIGEST_SIZE] = VmCommittedExe::<SC>::compute_exe_commit(
            &app_program_commit,
            app_exe,
            app_memory_config,
        )
        .into();
        let vm_commit: [Val<SC>; DIGEST_SIZE] = leaf_verifier_program_commit.into();
        Self::from_field_commit(exe_commit, vm_commit)
    }

    pub fn from_field_commit<F: PrimeField32>(
        exe_commit: [F; DIGEST_SIZE],
        vm_commit: [F; DIGEST_SIZE],
    ) -> Self {
        Self {
            app_exe_commit: CommitBytes::from_u32_digest(&exe_commit.map(|x| x.as_canonical_u32())),
            app_vm_commit: CommitBytes::from_u32_digest(&vm_commit.map(|x| x.as_canonical_u32())),
        }
    }
}

pub fn commit_app_exe(
    app_fri_params: FriParameters,
    app_exe: impl Into<VmExe<F>>,
) -> Arc<VmCommittedExe<SC>> {
    let exe: VmExe<_> = app_exe.into();
    let app_engine = BabyBearPoseidon2Engine::new(app_fri_params);
    Arc::new(VmCommittedExe::<SC>::commit(exe, app_engine.config().pcs()))
}

pub(crate) fn babybear_digest_to_bn254(digest: &[F; DIGEST_SIZE]) -> Bn254Fr {
    let mut ret = Bn254Fr::ZERO;
    let order = Bn254Fr::from_canonical_u32(BabyBear::ORDER_U32);
    let mut base = Bn254Fr::ONE;
    digest.iter().for_each(|&x| {
        ret += base * Bn254Fr::from_canonical_u32(x.as_canonical_u32());
        base *= order;
    });
    ret
}

fn bytes_to_bn254(bytes: &[u8; BN254_BYTES]) -> Bn254Fr {
    let order = Bn254Fr::from_canonical_u32(1 << 8);
    let mut ret = Bn254Fr::ZERO;
    let mut base = Bn254Fr::ONE;
    for byte in bytes.iter().rev() {
        ret += base * Bn254Fr::from_canonical_u8(*byte);
        base *= order;
    }
    ret
}

fn bn254_to_bytes(bn254: Bn254Fr) -> [u8; BN254_BYTES] {
    let mut ret = bn254.value.to_bytes();
    ret.reverse();
    ret
}

fn bytes_to_u32_digest(bytes: &[u8; BN254_BYTES]) -> [u32; DIGEST_SIZE] {
    let mut bigint = BigUint::ZERO;
    for byte in bytes.iter() {
        bigint <<= 8;
        bigint += BigUint::from(*byte);
    }
    let order = BabyBear::ORDER_U32;
    from_fn(|_| {
        let bigint_digit = bigint.clone() % order;
        let digit = if bigint_digit == BigUint::ZERO {
            0u32
        } else {
            bigint_digit.to_u32_digits()[0]
        };
        bigint /= order;
        digit
    })
}

fn u32_digest_to_bytes(digest: &[u32; DIGEST_SIZE]) -> [u8; BN254_BYTES] {
    bn254_to_bytes(babybear_digest_to_bn254(&digest.map(F::from_canonical_u32)))
}
