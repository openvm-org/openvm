use std::sync::Arc;

use openvm_circuit::{
    arch::{instructions::exe::VmExe, VmConfig},
    system::program::trace::VmCommittedExe,
};
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{config::StarkGenericConfig, p3_field::PrimeField32};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::p3_field::FieldAlgebra,
    p3_baby_bear::BabyBear,
    p3_bn254_fr::Bn254Fr,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{types::BN254_BYTES, NonRootCommittedExe, F, SC};

/// `AppExecutionCommit` has all the commitments users should check against the final proof.
/// Each commit is stored as a u32 array, where each element is a member of F. The array
/// represents a little-endian base-F number.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppExecutionCommit {
    /// Commitment of the executable. It's computed as
    /// compress(
    ///     compress(
    ///         hash(app_program_commit),
    ///         hash(init_memory_commit)
    ///     ),
    ///     hash(right_pad(pc_start, 0))
    /// )
    /// `right_pad` example, if pc_start = 123, right_pad(pc_start, 0) = \[123,0,0,0,0,0,0,0\]
    pub exe_commit: [u32; DIGEST_SIZE],
    /// Commitment of the leaf VM verifier program which commits the VmConfig of App VM.
    /// Internal verifier will verify `leaf_vm_verifier_commit`.
    pub vm_commit: [u32; DIGEST_SIZE],
}

impl AppExecutionCommit {
    /// Users should use this function to compute `AppExecutionCommit` and check it against the
    /// final proof.
    pub fn compute<VC: VmConfig<F>>(
        app_vm_config: &VC,
        app_exe: &NonRootCommittedExe,
        leaf_vm_verifier_exe: &NonRootCommittedExe,
    ) -> Self {
        let exe_commit: [F; DIGEST_SIZE] = app_exe
            .compute_exe_commit(&app_vm_config.system().memory_config)
            .into();
        let vm_commit: [F; DIGEST_SIZE] = leaf_vm_verifier_exe.committed_program.commitment.into();
        Self::from_field_commit(exe_commit, vm_commit)
    }

    pub fn from_field_commit(exe_commit: [F; DIGEST_SIZE], vm_commit: [F; DIGEST_SIZE]) -> Self {
        Self {
            exe_commit: exe_commit.map(|x| x.as_canonical_u32()),
            vm_commit: vm_commit.map(|x| x.as_canonical_u32()),
        }
    }

    pub fn to_bytes(&self) -> AppExecutionCommitBytes {
        AppExecutionCommitBytes {
            app_exe_commit: self.exe_commit_to_bytes(),
            app_vm_commit: self.vm_commit_to_bytes(),
        }
    }

    pub fn exe_commit_to_bn254(&self) -> Bn254Fr {
        babybear_u32_digest_to_bn254(&self.exe_commit)
    }

    pub fn vm_commit_to_bn254(&self) -> Bn254Fr {
        babybear_u32_digest_to_bn254(&self.vm_commit)
    }

    pub fn exe_commit_to_bytes(&self) -> [u8; BN254_BYTES] {
        let mut ret = self.exe_commit_to_bn254().value.to_bytes();
        ret.reverse();
        ret
    }

    pub fn vm_commit_to_bytes(&self) -> [u8; BN254_BYTES] {
        let mut ret = self.vm_commit_to_bn254().value.to_bytes();
        ret.reverse();
        ret
    }
}

fn babybear_u32_digest_to_bn254(digest: &[u32; DIGEST_SIZE]) -> Bn254Fr {
    babybear_digest_to_bn254(&digest.map(F::from_canonical_u32))
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

pub fn commit_app_exe(
    app_fri_params: FriParameters,
    app_exe: impl Into<VmExe<F>>,
) -> Arc<NonRootCommittedExe> {
    let exe: VmExe<_> = app_exe.into();
    let app_engine = BabyBearPoseidon2Engine::new(app_fri_params);
    Arc::new(VmCommittedExe::<SC>::commit(exe, app_engine.config.pcs()))
}

/// Byte representation of AppExecutionCommit. Because the commits in AppExecutionCommit
/// are stored as an array representing a base-F number, the 256 bits that make up each
/// commit are different than actual the 256-bit representation of the Bn254. This class
/// stores the latter to enable consistent serialization between different proof outputs.
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct AppExecutionCommitBytes {
    #[serde_as(as = "serde_with::hex::Hex")]
    /// 1 Bn254Fr public value for app exe commit in big-endian bytes.
    pub app_exe_commit: [u8; BN254_BYTES],
    #[serde_as(as = "serde_with::hex::Hex")]
    /// 1 Bn254Fr public value for app vm commit in big-endian bytes.
    pub app_vm_commit: [u8; BN254_BYTES],
}

impl AppExecutionCommitBytes {
    pub fn exe_commit_to_bn254(&self) -> Bn254Fr {
        bytes_to_bn254(&self.app_exe_commit)
    }

    pub fn vm_commit_to_bn254(&self) -> Bn254Fr {
        bytes_to_bn254(&self.app_vm_commit)
    }
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
