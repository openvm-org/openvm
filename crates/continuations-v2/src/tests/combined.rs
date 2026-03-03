use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher,
        instructions::{exe::VmExe, NATIVE_AS},
        ContinuationVmProver, VirtualMachine, VmInstance,
    },
    system::memory::{
        dimensions::MemoryDimensions,
        merkle::{public_values::UserPublicValuesProof, MerkleTree},
    },
};
use openvm_cuda_backend::{prelude::SC, BabyBearPoseidon2GpuEngine};
use openvm_rv32im_circuit::Rv32ImBuilder;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::{proof::Proof, AirRef, PartitionedBaseAir, StarkEngine};
use openvm_stark_sdk::{
    config::{baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine, baby_bear_poseidon2::F},
    utils::setup_tracing_with_log_level,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::prelude::DIGEST_SIZE;
use tracing::{warn, Level};
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use super::{
    app_system_params, expected_deferral_leaf_io_commit,
    generate_deferral_internal_recursive_proof_from_copies, internal_system_params,
    leaf_system_params, root_system_params, test_rv32im_config,
};
use crate::{
    circuit::{
        deferral::{verify::DeferralMerkleProofs, DeferralCircuitPvs},
        inner::ProofsType,
    },
    prover::{
        ChildVkKind, DeferralHookGpuProver as DeferralHookProver,
        DeferralInnerGpuProver as DeferralInnerProver,
        DeferralVerifyGpuProver as DeferralVerifyProver, InnerGpuProver as InnerProver,
        RootCpuProver as RootProver,
    },
};

type Engine = BabyBearPoseidon2GpuEngine;
type RootEngine = BabyBearBn254Poseidon2CpuEngine;

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
        DeferralInnerProver::<2>::new::<Engine>(deferral_vk, leaf_system_params(), false);
    let internal_0_prover = DeferralInnerProver::<2>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let internal_1_prover = DeferralInnerProver::<2>::new::<Engine>(
        internal_0_prover.get_vk(),
        internal_system_params(),
        true,
    );
    let hook_prover =
        DeferralHookProver::new::<Engine>(internal_1_prover.get_vk(), root_system_params());

    hook_prover.get_cached_commit()
}

fn read_def_pvs(proof: &Proof<SC>) -> DeferralPvs<F> {
    *proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow()
}

fn make_absent_trace_pvs(
    deferral_hook_final_hash: [F; DIGEST_SIZE],
    depth: F,
) -> (DeferralPvs<F>, bool) {
    (
        DeferralPvs {
            initial_acc_hash: deferral_hook_final_hash,
            final_acc_hash: deferral_hook_final_hash,
            depth,
        },
        false,
    )
}

fn generate_unset_merkle_proof(
    memory_dimensions: MemoryDimensions,
    merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
) -> Vec<[F; DIGEST_SIZE]> {
    let block_id = 1u32; // second digest in NATIVE_AS
    let leaf_idx = (1u64 << memory_dimensions.overall_height())
        + memory_dimensions.label_to_index((NATIVE_AS, block_id));

    let mut node_idx = leaf_idx;
    let mut proof = Vec::with_capacity(memory_dimensions.overall_height());
    while node_idx > 1 {
        let sibling_idx = if node_idx.is_multiple_of(2) {
            node_idx + 1
        } else {
            node_idx - 1
        };
        proof.push(merkle_tree.get_node(sibling_idx));
        node_idx >>= 1;
    }

    proof
}

#[test]
fn test_vm_deferral_mix_combined_flow() -> Result<()> {
    setup_tracing_with_log_level(Level::WARN);

    // SECTION 0: Generate a def_hook_commit.
    let def_hook_commit = def_hook_commit();

    // SECTION 1: Create a regular VM internal-recursive proof + vk using that def_hook_commit.
    let config = test_rv32im_config();
    let system_config = config.rv32i.system.clone();
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

    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let initial_address_map = &instance.state().as_ref().unwrap().memory.memory;
    let initial_merkle_tree = MerkleTree::from_memory(
        initial_address_map,
        &memory_dimensions,
        &vm_poseidon2_hasher::<F>(),
    );
    let initial_merkle_proof = generate_unset_merkle_proof(memory_dimensions, &initial_merkle_tree);

    warn!("proving app proof");
    let app_proof = instance.prove(vec![input])?;

    let final_address_map = &instance.state().as_ref().unwrap().memory.memory;
    let final_merkle_tree = MerkleTree::from_memory(
        final_address_map,
        &memory_dimensions,
        &vm_poseidon2_hasher::<F>(),
    );
    let final_merkle_proof = generate_unset_merkle_proof(memory_dimensions, &final_merkle_tree);

    let merkle_proofs = DeferralMerkleProofs {
        initial_merkle_proof,
        final_merkle_proof,
    };

    let leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_pk.get_vk()),
        leaf_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving VM leaf aggregation proof");
    let leaf_vm_proof =
        leaf_prover.agg_prove_no_def::<Engine>(&app_proof.per_segment, ChildVkKind::App)?;
    let user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F> = app_proof.user_public_values;

    let internal_for_leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving VM internal-for-leaf aggregation proof");
    let internal_for_leaf_vm_proof = internal_for_leaf_prover
        .agg_prove_no_def::<Engine>(&[leaf_vm_proof], ChildVkKind::Standard)?;

    let internal_recursive_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
        Some(def_hook_commit),
    );
    warn!("proving VM internal-recursive aggregation proof");
    let internal_recursive_vm_proof = internal_recursive_prover
        .agg_prove_no_def::<Engine>(&[internal_for_leaf_vm_proof], ChildVkKind::Standard)?;

    let vm_internal_recursive_vk = internal_recursive_prover.get_vk();
    let vm_internal_recursive_pcs_data = internal_recursive_prover.get_self_vk_pcs_data().unwrap();

    // SECTION 1.5: Check that root prover with def_hook_commit verifies.
    let vm_root_prover = RootProver::new::<RootEngine>(
        vm_internal_recursive_vk.clone(),
        vm_internal_recursive_pcs_data.commitment.into(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        Some(def_hook_commit.into()),
        None,
    );
    let ctx = vm_root_prover.generate_proving_ctx_with_deferrals(
        internal_recursive_vm_proof.clone(),
        &user_pvs_proof,
        &merkle_proofs,
    );
    warn!("testing VM root prover on proof with unset deferral pvs (not part of mixed flow)");
    let root_proof = vm_root_prover.root_prove_from_ctx::<RootEngine>(ctx.unwrap())?;

    let root_vk = vm_root_prover.get_vk();
    let engine = RootEngine::new(root_vk.inner.params.clone());
    engine.verify(&root_vk, &root_proof)?;

    // SECTION 2: Create a deferral hook proof of that VM proof.
    let deferred_verify_prover = DeferralVerifyProver::new::<Engine>(
        vm_internal_recursive_vk.clone(),
        vm_internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        Some(def_hook_commit),
    );
    warn!("proving deferral verify proof from VM internal-recursive proof");
    let deferral_verify_proof = deferred_verify_prover.prove::<Engine>(
        internal_recursive_vm_proof.clone(),
        &user_pvs_proof,
        Some(&merkle_proofs),
    )?;

    let (deferral_internal_recursive_vk, deferral_internal_recursive_proof) =
        generate_deferral_internal_recursive_proof_from_copies(
            deferred_verify_prover.get_vk(),
            deferral_verify_proof.clone(),
            1,
        )?;
    let (leaf_input_commit, leaf_output_commit) =
        expected_deferral_leaf_io_commit(&deferral_verify_proof);

    let deferral_hook_prover =
        DeferralHookProver::new::<Engine>(deferral_internal_recursive_vk, root_system_params());
    warn!("proving deferral hook proof");
    let deferral_hook_proof = deferral_hook_prover.prove::<Engine>(
        deferral_internal_recursive_proof,
        vec![(leaf_input_commit, leaf_output_commit)],
    )?;

    // SECTION 3: Assert the deferral_hook_prover cached commit equals def_hook_commit.
    assert_eq!(deferral_hook_prover.get_cached_commit(), def_hook_commit);

    // SECTION 4: Feed deferral_hook_proof back into the VM prover via Deferral path and wrap to
    // internal-recursive.
    let def_hook_pvs_vec = deferral_hook_proof.public_values[0].clone();
    let deferral_hook_pvs: &DeferralPvs<F> = def_hook_pvs_vec.as_slice().borrow();
    let deferral_leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        deferral_hook_prover.get_vk(),
        leaf_system_params(),
        false,
        Some(def_hook_commit),
    );

    warn!("proving VM deferral-path leaf proof");
    let leaf_deferral_proof = deferral_leaf_prover.agg_prove::<Engine>(
        &[deferral_hook_proof],
        ChildVkKind::App,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            deferral_hook_pvs.final_acc_hash,
            deferral_hook_pvs.depth,
        )),
    )?;

    let deferral_internal_for_leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        deferral_leaf_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    let leaf_def_pvs = read_def_pvs(&leaf_deferral_proof);
    warn!("proving VM deferral-path internal-for-leaf proof");
    let internal_for_leaf_deferral_proof = deferral_internal_for_leaf_prover.agg_prove::<Engine>(
        &[leaf_deferral_proof],
        ChildVkKind::Standard,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            leaf_def_pvs.final_acc_hash,
            leaf_def_pvs.depth,
        )),
    )?;

    let deferral_internal_recursive_prover = InnerProver::<MAX_NUM_PROOFS>::new::<Engine>(
        deferral_internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
        Some(def_hook_commit),
    );
    let internal_for_leaf_def_pvs = read_def_pvs(&internal_for_leaf_deferral_proof);
    warn!("proving VM deferral-path internal-recursive proof");
    let internal_recursive_deferral_proof = deferral_internal_recursive_prover
        .agg_prove::<Engine>(
            &[internal_for_leaf_deferral_proof],
            ChildVkKind::Standard,
            ProofsType::Deferral,
            Some(make_absent_trace_pvs(
                internal_for_leaf_def_pvs.final_acc_hash,
                internal_for_leaf_def_pvs.depth,
            )),
        )?;

    // SECTION 5: Aggregate VM internal-recursive proof + deferral internal-recursive proof via
    // Mixed pathway.
    warn!("proving mixed-path internal-recursive aggregation proof");
    let mixed_internal_recursive_proof = internal_recursive_prover.agg_prove::<Engine>(
        &[
            internal_recursive_vm_proof,
            internal_recursive_deferral_proof,
        ],
        ChildVkKind::RecursiveSelf,
        ProofsType::Mix,
        None,
    )?;

    // SECTION 6: Wrap once more using Combined pathway.
    warn!("proving combined-path wrapper proof");
    let combined_internal_recursive_proof = internal_recursive_prover.agg_prove::<Engine>(
        &[mixed_internal_recursive_proof],
        ChildVkKind::RecursiveSelf,
        ProofsType::Combined,
        None,
    )?;

    let combined_vk = internal_recursive_prover.get_vk();
    let engine = Engine::new(combined_vk.inner.params.clone());
    engine.verify(&combined_vk, &combined_internal_recursive_proof)?;

    Ok(())
}
