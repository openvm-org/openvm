extern crate core;

use std::{
    fs::{create_dir_all, read_to_string, write},
    path::Path,
    sync::Arc,
};

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
#[cfg(feature = "static-verifier")]
use axvm_native_recursion::halo2::verifier::Halo2VerifierCircuit;
use axvm_native_recursion::types::InnerConfig;
use axvm_transpiler::{elf::Elf, transpiler::Transpiler};
use config::{AggConfig, AppConfig};
use eyre::{eyre, Result};
use keygen::{AggProvingKey, AppProvingKey, AppVerifyingKey};
use p3_baby_bear::BabyBear;
use prover::generate_leaf_committed_exe;

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

    pub fn app_keygen<VC: VmConfig<F>>(&self, _config: AppConfig<VC>) -> Result<AppProvingKey<VC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    pub fn commit_app_exe<VC: VmConfig<F>>(
        &self,
        _config: AppConfig<VC>,
        _exe: AxVmExe<F>,
    ) -> Result<Arc<AxVmCommittedExe<SC>>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        todo!()
    }

    pub fn app_keygen_and_commit_exe<VC: VmConfig<F>>(
        &self,
        _config: AppConfig<VC>,
        _exe: AxVmExe<F>,
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

    pub fn agg_keygen<P: AsRef<Path>>(
        &self,
        config: AggConfig,
        output_path: Option<P>,
    ) -> Result<(AggConfig, AggProvingKey)> {
        let agg_pk: AggProvingKey = AggProvingKey::keygen(config);
        if let Some(output_path) = output_path {
            if let Some(parent) = output_path.as_ref().parent() {
                create_dir_all(parent)?;
            }
            println!("per_air length: {}", agg_pk.leaf_vm_pk.vm_pk.per_air.len());
            for i in 0..agg_pk.leaf_vm_pk.vm_pk.per_air.len() {
                let json =
                    serde_json::to_string(&agg_pk.leaf_vm_pk.vm_pk.per_air[i].preprocessed_data);
                if json.is_err() {
                    panic!("{:?}", json.err());
                }
                println!(
                    "leaf_vm_pk.vm_pk.per_air[{}].preprocessed_data: {}",
                    i,
                    json.unwrap()
                );
                let json = serde_json::to_string(&agg_pk.leaf_vm_pk.vm_pk.per_air[i].vk);
                if json.is_err() {
                    panic!("{:?}", json.err());
                }
                println!("leaf_vm_pk.vm_pk.per_air[{}].vk: {}", i, json.unwrap());
            }
            println!("Done!!");

            let json = serde_json::to_string(&agg_pk)?;
            if write(output_path, json).is_err() {
                return Err(eyre!("Failed to write aggregator proving key to file"));
            }
        }
        Ok((config, agg_pk))
    }

    pub fn load_agg_pk_from_file<P: AsRef<Path>>(
        &self,
        agg_pk_path: P,
    ) -> Result<(AggConfig, AggProvingKey)> {
        let (config, agg_pk) = serde_json::from_str(&read_to_string(agg_pk_path)?)?;
        Ok((config, agg_pk))
    }

    pub fn generate_leaf_committed_exe<VC: VmConfig<F>>(
        &self,
        config: AggConfig,
        app_pk: &AppProvingKey<VC>,
    ) -> Result<Arc<AxVmCommittedExe<SC>>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let leaf_exe = generate_leaf_committed_exe(&config, app_pk);
        Ok(leaf_exe)
    }

    pub fn agg_keygen_and_generate_leaf_committed_exe<VC: VmConfig<F>>(
        &self,
        config: AggConfig,
        app_pk: &AppProvingKey<VC>,
    ) -> Result<(AggProvingKey, Arc<AxVmCommittedExe<SC>>)>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let (_, agg_pk) = self.agg_keygen(config, None::<&Path>)?;
        let leaf_exe = self.generate_leaf_committed_exe(config, app_pk)?;
        Ok((agg_pk, leaf_exe))
    }

    #[cfg(feature = "static-verifier")]
    pub fn generate_static_verifier_circuit(
        &self,
        _agg_pk: AggProvingKey,
    ) -> Result<Halo2VerifierCircuit> {
        todo!()
    }

    #[cfg(feature = "static-verifier")]
    #[allow(clippy::too_many_arguments)]
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
    pub fn load_e2e_proof_from_file<P: AsRef<Path>>(&self, _e2e_proof_path: P) -> Result<EvmProof> {
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
