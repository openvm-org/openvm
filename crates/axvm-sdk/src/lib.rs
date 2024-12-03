extern crate core;

use std::{marker::PhantomData, sync::Arc};

use ax_stark_sdk::{
    ax_stark_backend::{prover::types::Proof, verifier::VerificationError, Chip},
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
    },
};
use axvm_circuit::{
    arch::{instructions::exe::AxVmExe, ExecutionError, VmConfig},
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_recursion::types::InnerConfig;
use axvm_transpiler::{elf::Elf, transpiler::Transpiler};
use config::AppConfig;
use keygen::AppProvingKey;
use p3_baby_bear::BabyBear;
#[cfg(feature = "static-verifier")]
use {
    axvm_native_recursion::halo2::verifier::Halo2VerifierCircuit, config::AggConfig,
    keygen::AggProvingKey, snark_verifier_sdk::Snark,
};

pub mod commit;
pub mod config;
pub mod e2e_prover;
#[cfg(feature = "static-verifier")]
pub mod static_verifier;

pub mod keygen;
pub mod prover;
pub mod verifier;

pub(crate) type SC = BabyBearPoseidon2Config;
pub(crate) type C = InnerConfig;
pub(crate) type F = BabyBear;
pub(crate) type OuterSC = BabyBearPoseidon2OuterConfig;

pub struct Sdk<VC: VmConfig<F>> {
    _marker: PhantomData<VC>,
}

impl<VC: VmConfig<F>> Sdk<VC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn build(_pkg_dir: &str) -> Elf {
        todo!()
    }

    pub fn transpile(_elf: Elf, _transpiler: Transpiler<F>) -> AxVmExe<F> {
        todo!()
    }

    pub fn execute(_inputs: Vec<Vec<F>>, _exe: AxVmExe<F>) -> Result<(), ExecutionError> {
        todo!()
    }

    pub fn app_keygen_and_commit_exe(
        _config: AppConfig<VC>,
        _exe: AxVmExe<F>,
        _output_dir: Option<&str>,
    ) -> (AppProvingKey<VC>, Arc<AxVmCommittedExe<SC>>) {
        todo!()
    }

    pub fn load_app_pk_from_cached_dir(
        _app_cache_dir: &str,
    ) -> (AppProvingKey<VC>, Arc<AxVmCommittedExe<SC>>) {
        todo!()
    }

    pub fn generate_app_proof(
        _inputs: Vec<Vec<F>>,
        _app_pk: AppProvingKey<VC>,
        _app_exe: Arc<AxVmCommittedExe<SC>>,
    ) -> Vec<Proof<SC>> {
        todo!()
    }

    pub fn verify_app_proof(
        _proof: Vec<Proof<SC>>,
        _app_pk: &AppProvingKey<VC>,
    ) -> Result<(), VerificationError> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn agg_keygen_and_commit_leaf_exe(
        _config: AggConfig,
        _output_dir: Option<&str>,
    ) -> (
        AggProvingKey,
        Arc<AxVmCommittedExe<SC>>,
        Halo2VerifierCircuit,
    ) {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn load_agg_pk_from_cached_dir(
        _agg_cache_dir: &str,
    ) -> (
        AggProvingKey,
        Arc<AxVmCommittedExe<SC>>,
        Halo2VerifierCircuit,
    ) {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn generate_e2e_proof(
        _inputs: Vec<Vec<F>>,
        _app_pk: AppProvingKey<VC>,
        _app_exe: Arc<AxVmCommittedExe<SC>>,
        _agg_pk: AggProvingKey,
        _leaf_exe: Arc<AxVmCommittedExe<SC>>,
        _static_verifier: Halo2VerifierCircuit,
    ) -> Snark {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn generate_snark_verifier_contract(_snark: Snark) -> Vec<u8> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    pub fn evm_verify_snark(_snark: Snark) -> Result<(), VerificationError> {
        todo!()
    }
}
