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

use crate::{NonRootCommittedExe, F, SC};

/// `AppExecutionCommit` has all the commitments users should check against the final proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppExecutionCommit {
    /// Commitment of the leaf VM verifier program which commits the VmConfig of App VM.
    /// Internal verifier will verify `leaf_vm_verifier_commit`.
    pub vm_commit: [u32; DIGEST_SIZE],
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

        Self {
            vm_commit: vm_commit.map(|x| x.as_canonical_u32()),
            exe_commit: exe_commit.map(|x| x.as_canonical_u32()),
        }
    }

    pub fn vm_commit_to_bn254(&self) -> Bn254Fr {
        babybear_u32_digest_to_bn254(&self.vm_commit)
    }

    pub fn exe_commit_to_bn254(&self) -> Bn254Fr {
        babybear_u32_digest_to_bn254(&self.exe_commit)
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
