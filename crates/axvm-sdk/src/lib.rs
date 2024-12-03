extern crate core;

use std::{path::Path, sync::Arc};

use ax_stark_sdk::{
    ax_stark_backend::{prover::types::Proof, verifier::VerificationError, Chip},
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
    },
};
use axvm_build::GuestOptions;
use axvm_circuit::{
    arch::{instructions::exe::AxVmExe, ExecutionError, VmConfig},
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_recursion::types::InnerConfig;
use axvm_transpiler::{elf::Elf, transpiler::Transpiler};
use config::AppConfig;
use eyre::Result;
use keygen::{AppProvingKey, AppVerifyingKey};
use p3_baby_bear::BabyBear;
#[cfg(feature = "static-verifier")]
use {
    axvm_native_recursion::halo2::verifier::Halo2VerifierCircuit, config::AggConfig,
    keygen::AggProvingKey,
};

pub mod commit;
pub mod config;
pub mod prover;
#[cfg(feature = "static-verifier")]
pub mod static_verifier;

pub mod keygen;
pub mod verifier;

mod io;
pub use io::*;

pub(crate) type SC = BabyBearPoseidon2Config;
pub(crate) type C = InnerConfig;
pub(crate) type F = BabyBear;
pub(crate) type OuterSC = BabyBearPoseidon2OuterConfig;

pub struct Sdk;

impl Sdk {
    pub fn build<P: AsRef<Path>>(&self, _guest_opts: GuestOptions, _pkg_dir: P) -> Result<Elf> {
        todo!()
    }

    pub fn transpile(&self, _elf: Elf, _transpiler: Transpiler<F>) -> Result<AxVmExe<F>> {
        todo!()
    }

    pub fn execute(&self, _exe: AxVmExe<F>, _inputs: StdIn) -> Result<(), ExecutionError> {
        todo!()
    }

    pub fn app_keygen_and_commit_exe<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        _config: AppConfig<VC>,
        _exe: AxVmExe<F>,
        _output_path: Option<P>,
    ) -> Result<(AppProvingKey<VC>, Arc<AxVmCommittedExe<SC>>)>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    pub fn load_app_pk_from_cached_dir<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        _app_cache_path: P,
    ) -> Result<(AppProvingKey<VC>, Arc<AxVmCommittedExe<SC>>)>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    pub fn generate_app_proof<VC: VmConfig<F>>(
        &self,
        _app_pk: AppProvingKey<VC>,
        _app_exe: Arc<AxVmCommittedExe<SC>>,
        _inputs: StdIn,
    ) -> Result<Vec<Proof<SC>>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    pub fn verify_app_proof(
        &self,
        _proof: Vec<Proof<SC>>,
        _app_vk: &AppVerifyingKey,
    ) -> Result<(), VerificationError> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn agg_keygen_and_commit_leaf_exe<P: AsRef<Path>>(
        &self,
        _config: AggConfig,
        _output_path: Option<P>,
    ) -> Result<(
        AggProvingKey,
        Arc<AxVmCommittedExe<SC>>,
        Halo2VerifierCircuit,
    )> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn load_agg_pk_from_cached_dir<P: AsRef<Path>>(
        &self,
        _agg_cache_path: P,
    ) -> Result<(
        AggProvingKey,
        Arc<AxVmCommittedExe<SC>>,
        Halo2VerifierCircuit,
    )> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn generate_e2e_proof<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        _app_pk: AppProvingKey<VC>,
        _app_exe: Arc<AxVmCommittedExe<SC>>,
        _agg_pk: AggProvingKey,
        _leaf_exe: Arc<AxVmCommittedExe<SC>>,
        _static_verifier: Halo2VerifierCircuit,
        _inputs: StdIn,
        _output_path: Option<P>,
    ) -> Result<EvmProof>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn load_e2e_proof_from_cached_dir<P: AsRef<Path>>(
        &self,
        _e2e_proof_path: P,
    ) -> Result<EvmProof> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn generate_snark_verifier_contract<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        _evm_proof: EvmProof,
        _app_pk: AppProvingKey<VC>,
        _agg_pk: AggProvingKey,
        _output_path: P,
    ) -> Result<()>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn evm_verify_snark<P: AsRef<Path>>(
        &self,
        _evm_proof: EvmProof,
        _contract_path: P,
    ) -> Result<(), VerificationError> {
        todo!()
    }
}
