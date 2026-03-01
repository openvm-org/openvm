use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{instructions::exe::VmExe, ContinuationVmProver, VirtualMachine, VmInstance},
    system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_cuda_backend::{prelude::SC, BabyBearPoseidon2GpuEngine};
use openvm_rv32im_circuit::Rv32ImBuilder;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::{proof::Proof, AirRef, PartitionedBaseAir, StarkEngine};
use openvm_stark_sdk::{config::baby_bear_poseidon2::F, utils::setup_tracing_with_log_level};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::prelude::DIGEST_SIZE;
use tracing::Level;
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use super::{
    app_system_params, expected_deferral_leaf_io_commit,
    generate_deferral_internal_recursive_proof_from_copies, internal_system_params,
    leaf_system_params, root_system_params, test_rv32im_config,
};
use crate::{
    circuit::{deferral::DeferralCircuitPvs, nonroot::ProofsType},
    prover::{
        ChildVkKind, DeferralNonRootGpuProver as DeferralNonRootProver,
        DeferralRootGpuProver as DeferralRootProver,
        DeferralVerifyGpuProver as DeferralVerifyProver, NonRootGpuProver as NonRootProver,
    },
};

type Engine = BabyBearPoseidon2GpuEngine;

const LOG_FIB_INPUT: usize = 10;
const MAX_NUM_PROOFS: usize = 4;

struct EmptyAirWithPvs(usize);

impl<F> BaseAir<F> for EmptyAirWithPvs {
    fn width(&self) -> usize {
        1
    }
}
impl<F> BaseAirWithPublicValues<F> for EmptyAirWithPvs {
    fn num_public_values(&self) -> usize {
        self.0
    }
}
impl<F> PartitionedBaseAir<F> for EmptyAirWithPvs {}
impl<AB: AirBuilder> Air<AB> for EmptyAirWithPvs {
    fn eval(&self, _builder: &mut AB) {}
}

fn def_hook_commit() -> [F; DIGEST_SIZE] {
    let engine = Engine::new(app_system_params());
    let (_, deferral_vk) = engine
        .keygen(&[Arc::new(EmptyAirWithPvs(DeferralCircuitPvs::<u8>::width())) as AirRef<SC>]);
    let deferral_vk = Arc::new(deferral_vk);

    let leaf_prover =
        DeferralNonRootProver::<2>::new::<Engine>(deferral_vk, leaf_system_params(), false);
    let internal_0_prover = DeferralNonRootProver::<2>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let internal_1_prover = DeferralNonRootProver::<2>::new::<Engine>(
        internal_0_prover.get_vk(),
        internal_system_params(),
        true,
    );
    let hook_prover =
        DeferralRootProver::new::<Engine>(internal_1_prover.get_vk(), root_system_params());

    hook_prover.get_cached_commit()
}

fn read_def_pvs(proof: &Proof<SC>) -> DeferralPvs<F> {
    *proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow()
}

fn make_absent_trace_pvs(
    deferral_root_final_hash: [F; DIGEST_SIZE],
    depth: F,
) -> (DeferralPvs<F>, bool) {
    (
        DeferralPvs {
            initial_acc_hash: deferral_root_final_hash,
            final_acc_hash: deferral_root_final_hash,
            depth,
        },
        false,
    )
}

#[test]
fn test_vm_deferral_mix_combined_flow() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);

    // SECTION 0: Generate a def_hook_commit.
    let def_hook_commit = def_hook_commit();

    // SECTION 1: Create a regular VM internal-recursive proof + vk using that def_hook_commit.
    let config = test_rv32im_config();
    let elf = Elf::decode(
        include_bytes!("../../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let input = (1u64 << LOG_FIB_INPUT)
        .to_le_bytes()
        .map(F::from_u8)
        .to_vec();

    let engine = Engine::new(app_system_params());
    let (vm, app_pk) = VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    let app_proof = instance.prove(vec![input])?;

    let leaf_prover = NonRootProver::<MAX_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_pk.get_vk()),
        leaf_system_params(),
        false,
        Some(def_hook_commit),
    );
    let leaf_vm_proof =
        leaf_prover.agg_prove_no_def::<Engine>(&app_proof.per_segment, ChildVkKind::App)?;
    let user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F> = app_proof.user_public_values;

    let internal_for_leaf_prover = NonRootProver::<MAX_NUM_PROOFS>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    let internal_for_leaf_vm_proof = internal_for_leaf_prover
        .agg_prove_no_def::<Engine>(&[leaf_vm_proof.clone()], ChildVkKind::Standard)?;

    let internal_recursive_prover = NonRootProver::<MAX_NUM_PROOFS>::new::<Engine>(
        internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
        Some(def_hook_commit),
    );
    let internal_recursive_vm_proof = internal_recursive_prover
        .agg_prove_no_def::<Engine>(&[internal_for_leaf_vm_proof.clone()], ChildVkKind::Standard)?;

    let vm_internal_recursive_vk = internal_recursive_prover.get_vk();
    let vm_internal_recursive_pcs_data = internal_recursive_prover.get_self_vk_pcs_data().unwrap();

    // SECTION 2: Create a deferral root proof of that VM proof.
    let system_config = test_rv32im_config().rv32i.system;
    let deferred_verify_prover = DeferralVerifyProver::new::<Engine>(
        vm_internal_recursive_vk.clone(),
        vm_internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
    );
    let deferral_verify_proof = deferred_verify_prover
        .prove::<Engine>(internal_recursive_vm_proof.clone(), &user_pvs_proof)?;

    let (deferral_internal_recursive_vk, deferral_internal_recursive_proof) =
        generate_deferral_internal_recursive_proof_from_copies(
            deferred_verify_prover.get_vk(),
            deferral_verify_proof.clone(),
            1,
        )?;
    let (leaf_input_commit, leaf_output_commit) =
        expected_deferral_leaf_io_commit(&deferral_verify_proof);

    let deferral_root_prover =
        DeferralRootProver::new::<Engine>(deferral_internal_recursive_vk, root_system_params());
    let deferral_root_proof = deferral_root_prover.prove::<Engine>(
        deferral_internal_recursive_proof,
        vec![(leaf_input_commit, leaf_output_commit)],
    )?;

    // SECTION 3: Assert the deferral_root_prover cached commit equals def_hook_commit.
    assert_eq!(deferral_root_prover.get_cached_commit(), def_hook_commit);

    // SECTION 4: Feed deferral_root_proof back into the VM prover via Deferral path and wrap to
    // internal-recursive.
    let deferral_root_pvs: &DeferralPvs<F> =
        deferral_root_proof.public_values[0].as_slice().borrow();
    let deferral_root_final_hash = deferral_root_pvs.final_acc_hash;

    let leaf_vm_def_pvs = read_def_pvs(&leaf_vm_proof);
    let internal_for_leaf_deferral_proof = internal_for_leaf_prover.agg_prove::<Engine>(
        &[leaf_vm_proof.clone()],
        ChildVkKind::Standard,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            deferral_root_final_hash,
            leaf_vm_def_pvs.depth,
        )),
    )?;

    let internal_for_leaf_def_pvs = read_def_pvs(&internal_for_leaf_deferral_proof);
    let internal_recursive_deferral_proof = internal_recursive_prover.agg_prove::<Engine>(
        &[internal_for_leaf_deferral_proof],
        ChildVkKind::Standard,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            deferral_root_final_hash,
            internal_for_leaf_def_pvs.depth,
        )),
    )?;

    // SECTION 5: Aggregate VM internal-recursive proof + deferral internal-recursive proof via
    // Mixed pathway.
    let mixed_path_prover = NonRootProver::<MAX_NUM_PROOFS>::new::<Engine>(
        vm_internal_recursive_vk,
        internal_system_params(),
        true,
        Some(def_hook_commit),
    );
    let mixed_internal_recursive_proof = mixed_path_prover.agg_prove::<Engine>(
        &[
            internal_recursive_vm_proof,
            internal_recursive_deferral_proof,
        ],
        ChildVkKind::Standard,
        ProofsType::Mix,
        None,
    )?;

    // SECTION 6: Wrap once more using Combined pathway.
    let combined_internal_recursive_proof = mixed_path_prover.agg_prove::<Engine>(
        &[mixed_internal_recursive_proof],
        ChildVkKind::RecursiveSelf,
        ProofsType::Combined,
        None,
    )?;

    let combined_vk = mixed_path_prover.get_vk();
    let engine = Engine::new(combined_vk.inner.params.clone());
    engine.verify(&combined_vk, &combined_internal_recursive_proof)?;

    Ok(())
}
