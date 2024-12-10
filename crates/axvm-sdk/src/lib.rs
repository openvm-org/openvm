extern crate core;

use std::{
    fs::{create_dir_all, read, write},
    io::Write,
    panic::catch_unwind,
    path::Path,
    sync::Arc,
};

use ax_stark_backend::engine::StarkEngine;
use ax_stark_sdk::{
    ax_stark_backend::{verifier::VerificationError, Chip},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_build::{build_guest_package, get_package, get_target_dir, GuestOptions};
use axvm_circuit::{
    arch::{instructions::exe::AxVmExe, ExecutionError, VmConfig},
    prover::ContinuationVmProof,
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_recursion::{
    halo2::{
        wrapper::{EvmVerifier, Halo2WrapperProvingKey},
        EvmProof,
    },
    types::InnerConfig,
};
use axvm_transpiler::{
    axvm_platform::memory::MEM_SIZE,
    elf::Elf,
    transpiler::{Transpiler, TranspilerError},
    FromElf,
};
use commit::commit_app_exe;
use config::AppConfig;
use eyre::{bail, Result};
use itertools::Itertools;
use keygen::AppProvingKey;

pub mod commit;
pub mod config;
pub mod prover;
pub mod static_verifier;

pub mod keygen;
pub mod verifier;

mod io;
pub use io::*;

use crate::{
    config::FullAggConfig,
    keygen::FullAggProvingKey,
    prover::{AppProver, ContinuationProver},
};

pub(crate) type SC = BabyBearPoseidon2Config;
pub(crate) type C = InnerConfig;
pub(crate) type F = BabyBear;
pub(crate) type OuterSC = BabyBearPoseidon2OuterConfig;
pub(crate) type NonRootCommittedExe = AxVmCommittedExe<SC>;

pub struct Sdk;

impl Sdk {
    pub fn build<P: AsRef<Path>>(&self, guest_opts: GuestOptions, pkg_dir: P) -> Result<Elf> {
        if guest_opts.use_docker.is_some() {
            bail!("docker build is not supported yet");
        }
        let pkg = get_package(pkg_dir.as_ref());
        let target_dir = get_target_dir(&pkg.manifest_path);
        if let Err(Some(code)) =
            build_guest_package(&pkg, target_dir.clone(), &guest_opts.into(), None)
        {
            return Err(eyre::eyre!("Failed to build guest: code = {}", code));
        }
        eprintln!("target_dir: {:?}", target_dir);
        eprintln!("targets: {:?}", pkg.targets);

        let elf_path = pkg
            .targets
            .into_iter()
            .filter(|target| target.kind.iter().any(|kind| kind == "bin"))
            .exactly_one()
            .map(|target| {
                target_dir
                    .join("riscv32im-risc0-zkvm-elf")
                    .join("release")
                    .join(&target.name)
            })?;
        let data = read(elf_path)?;
        Elf::decode(&data, MEM_SIZE as u32)
    }

    pub fn transpile(
        &self,
        elf: Elf,
        transpiler: Transpiler<F>,
    ) -> Result<AxVmExe<F>, TranspilerError> {
        AxVmExe::from_elf(elf, transpiler)
    }

    pub fn execute(&self, _exe: AxVmExe<F>, _inputs: StdIn) -> Result<(), ExecutionError> {
        todo!()
    }

    pub fn commit_app_exe(
        &self,
        app_fri_params: FriParameters,
        exe: AxVmExe<F>,
    ) -> Result<Arc<NonRootCommittedExe>> {
        Ok(commit_app_exe(app_fri_params, exe))
    }

    pub fn app_keygen<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        config: AppConfig<VC>,
        output_path: Option<P>,
    ) -> Result<AppProvingKey<VC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_pk = AppProvingKey::keygen(config);
        if let Some(output_path) = output_path {
            if let Some(parent) = output_path.as_ref().parent() {
                create_dir_all(parent)?;
            }
            let output: Vec<u8> = bson::to_vec(&app_pk)?;
            write(output_path, output)?;
        }
        Ok(app_pk)
    }
    pub fn load_app_pk_from_file<VC: VmConfig<F>, P: AsRef<Path>>(
        &self,
        app_pk_path: P,
    ) -> Result<AppProvingKey<VC>> {
        let ret = bson::from_reader(std::fs::File::open(app_pk_path)?)?;
        Ok(ret)
    }

    pub fn create_app_prover<VC: VmConfig<F>>(
        &self,
        app_vm_pk: AppProvingKey<VC>,
        app_committed_exe: Arc<NonRootCommittedExe>,
    ) -> Result<AppProver<VC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        Ok(AppProver::new(app_vm_pk.app_vm_pk, app_committed_exe))
    }

    pub fn verify_app_proof<VC: VmConfig<F>>(
        &self,
        app_pk: &AppProvingKey<VC>,
        proof: &ContinuationVmProof<SC>,
    ) -> Result<(), VerificationError> {
        let e = BabyBearPoseidon2Engine::new(app_pk.app_vm_pk.fri_params);
        for seg_proof in &proof.per_segment {
            e.verify(&app_pk.app_vm_pk.vm_pk.get_vk(), seg_proof)?
        }
        // TODO: verify continuation.
        Ok(())
    }

    pub fn agg_keygen<P: AsRef<Path>>(
        &self,
        config: FullAggConfig,
        output_path: Option<P>,
    ) -> Result<FullAggProvingKey> {
        let agg_pk = FullAggProvingKey::keygen(config);
        if let Some(output_path) = output_path {
            if let Some(parent) = output_path.as_ref().parent() {
                create_dir_all(parent)?;
            }
            let output: Vec<u8> = bson::to_vec(&agg_pk)?;
            write(output_path, output)?;
        }
        Ok(agg_pk)
    }

    pub fn load_agg_pk_from_file<P: AsRef<Path>>(
        &self,
        agg_pk_path: P,
    ) -> Result<FullAggProvingKey> {
        let ret = bson::from_reader(std::fs::File::open(agg_pk_path)?)?;
        Ok(ret)
    }

    pub fn create_e2e_prover<VC: VmConfig<F>>(
        &self,
        app_pk: AppProvingKey<VC>,
        app_exe: Arc<NonRootCommittedExe>,
        agg_pk: FullAggProvingKey,
    ) -> Result<ContinuationProver<VC>> {
        Ok(ContinuationProver::new(app_pk, app_exe, agg_pk))
    }

    pub fn load_evm_proof_from_file<P: AsRef<Path>>(&self, evm_proof_path: P) -> Result<EvmProof> {
        let ret = bson::from_reader(std::fs::File::open(evm_proof_path)?)?;
        Ok(ret)
    }

    pub fn generate_snark_verifier_contract<P: AsRef<Path>>(
        &self,
        full_agg_proving_key: &FullAggProvingKey,
        output_path: Option<P>,
    ) -> Result<EvmVerifier> {
        let evm_verifier = full_agg_proving_key
            .halo2_pk
            .wrapper
            .generate_evm_verifier();
        if let Some(output_path) = output_path {
            let mut f = std::fs::File::create(output_path)?;
            f.write_all(&bson::to_vec(&evm_verifier)?)?;
        }
        Ok(evm_verifier)
    }

    pub fn load_snark_verifier_contract<P: AsRef<Path>>(&self, path: P) -> Result<EvmVerifier> {
        let ret = bson::from_reader(std::fs::File::open(path)?)?;
        Ok(ret)
    }

    pub fn verify_evm_proof(&self, evm_verifier: &EvmVerifier, evm_proof: &EvmProof) -> bool {
        // FIXME: we should return the concrete error.
        catch_unwind(|| {
            Halo2WrapperProvingKey::evm_verify(evm_verifier, evm_proof);
        })
        .is_ok()
    }
}
