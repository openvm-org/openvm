use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        deferral::{DeferralResult, DeferralState},
        hasher::poseidon2::vm_poseidon2_hasher,
        instructions::{exe::VmExe, DEFERRAL_AS},
        ContinuationVmProver, Streams, VirtualMachine, VmInstance,
    },
    system::memory::{
        dimensions::MemoryDimensions,
        merkle::{public_values::UserPublicValuesProof, MerkleTree},
    },
    utils::test_utils::test_system_config,
};
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
use openvm_deferral_circuit::{
    DeferralCpuBuilder, DeferralExtension, DeferralFn, Rv32DeferralConfig,
};
use openvm_deferral_transpiler::DeferralTranspilerExtension;
use openvm_recursion_circuit::{
    prelude::DIGEST_SIZE,
    utils::{poseidon2_hash_slice, poseidon2_hash_slice_with_states},
};
use openvm_rv32im_circuit::{Rv32I, Rv32Io, Rv32M};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::{proof::Proof, AirRef, StarkEngine};
use openvm_stark_sdk::{
    config::{
        baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
        baby_bear_poseidon2::{
            poseidon2_compress_with_capacity, BabyBearPoseidon2CpuEngine, DuplexSponge, F,
        },
    },
    utils::setup_tracing_with_log_level,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use openvm_verify_stark_host::pvs::{DeferralPvs, DEF_PVS_AIR_ID};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use tracing::{warn, Level};

use super::{
    app_system_params,
    dummy::{generate_dummy_def_proof, EmptyAirWithPvs},
    internal_system_params, leaf_system_params, root_system_params,
};
use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::{DeferralCircuitPvs, DeferralMerkleProofs, DEF_HOOK_PVS_AIR_ID},
        inner::ProofsType,
    },
    prover::{
        ChildVkKind, DeferralChildVkKind, DeferralHookGpuProver as DeferralHookProver,
        DeferralInnerGpuProver as DeferralInnerProver, InnerGpuProver as InnerProver,
        RootCpuProver as RootProver,
    },
    SC,
};

type GpuEngine = BabyBearPoseidon2GpuEngine;
type AppEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
type RootEngine = BabyBearBn254Poseidon2CpuEngine;

const NUM_DEF_CIRCUITS: usize = 3;
const MAX_NUM_PROOFS: usize = 4;

const INPUT_COMMIT_0: [u8; 32] = [0x11; 32];
const INPUT_COMMIT_1: [u8; 32] = [0x22; 32];
const INPUT_COMMIT_2: [u8; 32] = [0x33; 32];

const INPUT_RAW_0: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const INPUT_RAW_1: [u8; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
const INPUT_RAW_2: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

fn make_deferral_extension(commits: Vec<[u8; 32]>) -> DeferralExtension {
    let fns: Vec<_> = (0..NUM_DEF_CIRCUITS)
        .map(|idx| {
            Arc::new(DeferralFn::new(move |input_raw| {
                let mut prefix_sum = 0u16;
                input_raw
                    .iter()
                    .map(|&byte| {
                        prefix_sum += byte as u16;
                        (prefix_sum + idx as u16) as u8
                    })
                    .collect()
            }))
        })
        .collect();
    DeferralExtension::new(fns, commits)
}

fn deferral_fn_output(idx: u16, input_raw: &[u8]) -> Vec<u8> {
    let mut prefix_sum = 0u16;
    input_raw
        .iter()
        .map(|&byte| {
            prefix_sum += byte as u16;
            (prefix_sum + idx) as u8
        })
        .collect()
}

fn input_commit_to_f(commit: &[u8; 32]) -> [F; DIGEST_SIZE] {
    std::array::from_fn(|i| {
        let base = i * 4;
        F::new(
            commit[base] as u32
                | (commit[base + 1] as u32) << 8
                | (commit[base + 2] as u32) << 16
                | (commit[base + 3] as u32) << 24,
        )
    })
}

fn compute_output_f_commit(deferral_idx: u32, output_raw: &[u8]) -> [F; DIGEST_SIZE] {
    assert!(output_raw.len().is_multiple_of(DIGEST_SIZE));
    let mut state = [F::ZERO; DIGEST_SIZE];
    state[0] = F::new(deferral_idx);
    for chunk in output_raw.chunks_exact(DIGEST_SIZE) {
        let bytes: [F; DIGEST_SIZE] = std::array::from_fn(|i| F::from_u8(chunk[i]));
        state = poseidon2_compress_with_capacity(state, bytes).0;
    }
    state
}

fn hash_deferral_commit(commit: [F; DIGEST_SIZE]) -> [F; DIGEST_SIZE] {
    poseidon2_compress_with_capacity(commit, [F::ZERO; DIGEST_SIZE]).0
}

fn merkle_root_from_commits(commits: &[[F; DIGEST_SIZE]]) -> [F; DIGEST_SIZE] {
    assert!(commits.len().is_power_of_two(),);
    assert!(!commits.is_empty());

    let mut layer: Vec<[F; DIGEST_SIZE]> = commits
        .iter()
        .map(|commit| hash_deferral_commit(*commit))
        .collect();

    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|pair| poseidon2_compress_with_capacity(pair[0], pair[1]).0)
            .collect();
    }

    layer[0]
}

fn apply_onion_updates_for_circuit(
    commits: &mut [[F; DIGEST_SIZE]],
    def_idx: usize,
    io_commits: &[([F; DIGEST_SIZE], [F; DIGEST_SIZE])],
) {
    let input_idx = 2 * def_idx;
    let output_idx = input_idx + 1;
    for (input_commit, output_commit) in io_commits {
        commits[input_idx] = poseidon2_compress_with_capacity(commits[input_idx], *input_commit).0;
        commits[output_idx] =
            poseidon2_compress_with_capacity(commits[output_idx], *output_commit).0;
    }
}

fn generate_set_merkle_proof(
    memory_dimensions: MemoryDimensions,
    merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
    start_depth: usize,
    expected_start_node: [F; DIGEST_SIZE],
) -> Vec<[F; DIGEST_SIZE]> {
    let leaf_idx = (1u64 << memory_dimensions.overall_height())
        + memory_dimensions.label_to_index((DEFERRAL_AS, 0));
    assert_eq!(leaf_idx & ((1u64 << start_depth) - 1), 0);

    let mut node_idx = leaf_idx >> start_depth;
    assert_eq!(merkle_tree.get_node(node_idx), expected_start_node);

    let mut proof = Vec::with_capacity(memory_dimensions.overall_height());
    proof.extend(std::iter::repeat_n([F::ZERO; DIGEST_SIZE], start_depth));
    while node_idx > 1 {
        let sibling_idx = if node_idx.is_multiple_of(2) {
            node_idx + 1
        } else {
            node_idx - 1
        };
        proof.push(merkle_tree.get_node(sibling_idx));
        node_idx >>= 1;
    }
    assert_eq!(proof.len(), memory_dimensions.overall_height());

    proof
}

fn read_def_pvs(proof: &Proof<SC>) -> DeferralPvs<F> {
    *proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow()
}

fn read_hook_pvs(proof: &Proof<SC>) -> DeferralPvs<F> {
    *proof.public_values[DEF_HOOK_PVS_AIR_ID].as_slice().borrow()
}

fn make_absent_trace_pvs(
    initial_acc_hash: [F; DIGEST_SIZE],
    final_acc_hash: [F; DIGEST_SIZE],
    depth: F,
    is_right: bool,
) -> (DeferralPvs<F>, bool) {
    (
        DeferralPvs {
            initial_acc_hash,
            final_acc_hash,
            depth,
        },
        is_right,
    )
}

#[test]
fn test_deferral_e2e() -> Result<()> {
    setup_tracing_with_log_level(Level::WARN);

    // =========================================================================
    // SECTION 0: Compute def_hook_commit via the deferral aggregation chain.
    //
    // All three deferral circuits use EmptyAirWithPvs.
    // =========================================================================
    let gpu_engine = GpuEngine::new(app_system_params());
    let empty_air = Arc::new(EmptyAirWithPvs(DeferralCircuitPvs::<u8>::width())) as AirRef<SC>;
    let (_, def_circuit_vk) = gpu_engine.keygen(&[empty_air]);
    let def_circuit_vk = Arc::new(def_circuit_vk);

    let def_leaf_prover =
        DeferralInnerProver::new::<GpuEngine>(def_circuit_vk.clone(), leaf_system_params(), false);
    let def_i0_prover = DeferralInnerProver::new::<GpuEngine>(
        def_leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let def_i1_prover = DeferralInnerProver::new::<GpuEngine>(
        def_i0_prover.get_vk(),
        internal_system_params(),
        true,
    );
    let hook_prover_for_commit =
        DeferralHookProver::new::<GpuEngine>(def_i1_prover.get_vk(), root_system_params());
    let def_hook_commit = hook_prover_for_commit.get_cached_commit();

    // Compute vk commit using [cached_commit, vk_pre_hash] for def/leaf/i4l.
    let def_circuit_dag_commit = def_leaf_prover.get_dag_commit(false);
    let def_leaf_dag_commit = def_i0_prover.get_dag_commit(false);
    let def_i4l_dag_commit = def_i1_prover.get_dag_commit(false);

    let def_vk_commit = poseidon2_hash_slice_with_states(
        &[
            def_circuit_dag_commit.cached_commit,
            def_circuit_dag_commit.vk_pre_hash,
            def_leaf_dag_commit.cached_commit,
            def_leaf_dag_commit.vk_pre_hash,
            def_i4l_dag_commit.cached_commit,
            def_i4l_dag_commit.vk_pre_hash,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>(),
    )
    .0;

    let def_vk_commit_bytes = def_vk_commit
        .iter()
        .flat_map(|f| f.to_unique_u32().to_le_bytes())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let transpiler_commits = vec![def_vk_commit_bytes; NUM_DEF_CIRCUITS];

    // =========================================================================
    // SECTION 1: Set up Rv32DeferralConfig, build ELF, set up deferral streams.
    // =========================================================================
    let mut system = test_system_config().with_max_segment_len(1 << 20);
    system.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;

    let config = Rv32DeferralConfig {
        system: system.clone(),
        rv32i: Rv32I,
        rv32m: Rv32M::default(),
        io: Rv32Io,
        deferral: make_deferral_extension(transpiler_commits.clone()),
    };

    // let elf = Elf::decode(
    //     include_bytes!("../../programs/examples/multiple.elf"),
    //     MEM_SIZE as u32,
    // )?;
    let elf = openvm_toolchain_tests::build_example_program_at_path(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("programs"),
        "multiple",
        &config,
    )?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(DeferralTranspilerExtension::new(transpiler_commits)),
    )?;

    let in_commit_0_f = input_commit_to_f(&INPUT_COMMIT_0);
    let in_commit_1_f = input_commit_to_f(&INPUT_COMMIT_1);
    let in_commit_2_f = input_commit_to_f(&INPUT_COMMIT_2);

    let in_commit_0_hashed = poseidon2_hash_slice(&in_commit_0_f).0;
    let in_commit_1_hashed = poseidon2_hash_slice(&in_commit_1_f).0;
    let in_commit_2_hashed = poseidon2_hash_slice(&in_commit_2_f).0;

    let in_commit_0_bytes: CommitBytes = in_commit_0_hashed.into();
    let in_commit_1_bytes: CommitBytes = in_commit_1_hashed.into();
    let in_commit_2_bytes: CommitBytes = in_commit_2_hashed.into();

    // idx 0: unused, idx 1: 3 calls, idx 2: 1 call
    let state_unused = DeferralState::new(Vec::<DeferralResult>::new());
    let mut state1 = DeferralState::new(Vec::<DeferralResult>::new());
    state1.store_input(in_commit_0_bytes.as_slice().to_vec(), INPUT_RAW_0.to_vec());
    state1.store_input(in_commit_1_bytes.as_slice().to_vec(), INPUT_RAW_1.to_vec());
    state1.store_input(in_commit_2_bytes.as_slice().to_vec(), INPUT_RAW_2.to_vec());
    let mut state2 = DeferralState::new(Vec::<DeferralResult>::new());
    state2.store_input(in_commit_0_bytes.as_slice().to_vec(), INPUT_RAW_0.to_vec());

    let streams = Streams {
        input_stream: vec![in_commit_0_bytes, in_commit_1_bytes, in_commit_2_bytes]
            .into_iter()
            .map(|c| c.as_slice().iter().copied().map(F::from_u8).collect_vec())
            .collect_vec()
            .into(),
        deferrals: vec![state_unused, state1, state2],
        ..Default::default()
    };

    // =========================================================================
    // SECTION 2: Run the VM (CPU engine for DeferralCpuBuilder), capture merkle
    // proofs before and after execution.
    // =========================================================================
    let app_engine = AppEngine::new(app_system_params());
    let (vm, app_pk) = VirtualMachine::new_with_keygen(app_engine, DeferralCpuBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;

    let memory_dimensions = system.memory_config.memory_dimensions();
    let initial_address_map = &instance.state().as_ref().unwrap().memory.memory;
    let initial_merkle_tree = MerkleTree::from_memory(
        initial_address_map,
        &memory_dimensions,
        &vm_poseidon2_hasher::<F>(),
    );

    warn!("proving app proof (CPU)");
    let app_proof = instance.prove(streams)?;

    let final_address_map = &instance.state().as_ref().unwrap().memory.memory;
    let final_merkle_tree = MerkleTree::from_memory(
        final_address_map,
        &memory_dimensions,
        &vm_poseidon2_hasher::<F>(),
    );
    let user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F> = app_proof.user_public_values;

    // =========================================================================
    // SECTION 3: Generate dummy deferral circuit proofs using EmptyAirWithPvs.
    //
    // For each deferred_compute call, produce a proof whose DeferralCircuitPvs
    // carries the matching (input_commit, output_commit).
    // =========================================================================
    let cpu_engine = AppEngine::new(app_system_params());
    let (def_pk, _) = cpu_engine
        .keygen(&[Arc::new(EmptyAirWithPvs(DeferralCircuitPvs::<u8>::width())) as AirRef<SC>]);

    // Circuit idx 1: 3 calls (INPUT_COMMIT_0, INPUT_COMMIT_1, INPUT_COMMIT_2)
    let idx1_io: Vec<([F; DIGEST_SIZE], [F; DIGEST_SIZE], [F; DIGEST_SIZE])> = [
        (in_commit_0_f, in_commit_0_hashed, INPUT_RAW_0.as_slice()),
        (in_commit_1_f, in_commit_1_hashed, INPUT_RAW_1.as_slice()),
        (in_commit_2_f, in_commit_2_hashed, INPUT_RAW_2.as_slice()),
    ]
    .iter()
    .map(|(input_f, hashed, raw)| {
        let output_raw = deferral_fn_output(1, raw);
        let output_f = compute_output_f_commit(1, &output_raw);
        (*input_f, *hashed, output_f)
    })
    .collect();

    // Circuit idx 2: 1 call (INPUT_COMMIT_0)
    let idx2_io: Vec<([F; DIGEST_SIZE], [F; DIGEST_SIZE], [F; DIGEST_SIZE])> = {
        let output_raw = deferral_fn_output(2, &INPUT_RAW_0);
        let output_f = compute_output_f_commit(2, &output_raw);
        vec![(in_commit_0_f, in_commit_0_hashed, output_f)]
    };

    warn!("generating dummy deferral circuit proofs");
    let idx1_proofs: Vec<Proof<SC>> = idx1_io
        .iter()
        .map(|(inp, _, out)| generate_dummy_def_proof(&cpu_engine, &def_pk, *inp, *out))
        .collect();
    let idx2_proofs: Vec<Proof<SC>> = idx2_io
        .iter()
        .map(|(inp, _, out)| generate_dummy_def_proof(&cpu_engine, &def_pk, *inp, *out))
        .collect();

    let idx1_io = idx1_io.into_iter().map(|(_, x, y)| (x, y)).collect_vec();
    let idx2_io = idx2_io.into_iter().map(|(_, x, y)| (x, y)).collect_vec();

    let mut initial_commits = Vec::with_capacity(2 * NUM_DEF_CIRCUITS);
    for _ in 0..NUM_DEF_CIRCUITS {
        initial_commits.push(def_vk_commit);
        initial_commits.push([F::ZERO; DIGEST_SIZE]);
    }
    let mut final_commits = initial_commits.clone();
    apply_onion_updates_for_circuit(&mut final_commits, 1, &idx1_io);
    apply_onion_updates_for_circuit(&mut final_commits, 2, &idx2_io);

    // =========================================================================
    // SECTION 4: Aggregate dummy proofs into def hook proofs (one per circuit).
    //
    // For each circuit: leaf -> internal-for-leaf -> internal-recursive -> hook.
    // =========================================================================
    fn aggregate_deferral_tree(
        def_circuit_vk: &Arc<openvm_stark_backend::keygen::types::MultiStarkVerifyingKey<SC>>,
        proofs: Vec<Proof<SC>>,
        leaf_children: Vec<([F; DIGEST_SIZE], [F; DIGEST_SIZE])>,
    ) -> Result<Proof<SC>> {
        let leaf_prover = DeferralInnerProver::new::<GpuEngine>(
            def_circuit_vk.clone(),
            leaf_system_params(),
            false,
        );

        let mut current_proofs = proofs;
        let mut child_merkle_depth = 0usize;
        // Leaf layer
        let mut next = Vec::with_capacity(current_proofs.len().div_ceil(2));
        let layer_merkle_depth = if current_proofs.len() == 1 {
            None
        } else {
            Some(child_merkle_depth)
        };
        for chunk in current_proofs.chunks(2) {
            let kind = DeferralChildVkKind::DeferralCircuit;
            let proof = leaf_prover.agg_prove::<GpuEngine>(chunk, kind, layer_merkle_depth)?;
            next.push(proof);
        }
        current_proofs = next;
        child_merkle_depth += 1;

        let i4l_prover = DeferralInnerProver::new::<GpuEngine>(
            leaf_prover.get_vk(),
            internal_system_params(),
            false,
        );
        let mut next = Vec::with_capacity(current_proofs.len().div_ceil(2));
        let layer_merkle_depth = if current_proofs.len() == 1 {
            None
        } else {
            Some(child_merkle_depth)
        };
        for chunk in current_proofs.chunks(2) {
            let proof = i4l_prover.agg_prove::<GpuEngine>(
                chunk,
                DeferralChildVkKind::DeferralAggregation,
                layer_merkle_depth,
            )?;
            next.push(proof);
        }
        current_proofs = next;
        child_merkle_depth += 1;

        let ir_prover = DeferralInnerProver::new::<GpuEngine>(
            i4l_prover.get_vk(),
            internal_system_params(),
            true,
        );
        loop {
            let mut next = Vec::with_capacity(current_proofs.len().div_ceil(2));
            let layer_merkle_depth = if current_proofs.len() == 1 {
                None
            } else {
                Some(child_merkle_depth)
            };
            for chunk in current_proofs.chunks(2) {
                let proof = ir_prover.agg_prove::<GpuEngine>(
                    chunk,
                    DeferralChildVkKind::DeferralAggregation,
                    layer_merkle_depth,
                )?;
                next.push(proof);
            }
            current_proofs = next;
            child_merkle_depth += 1;

            if current_proofs.len() == 1 {
                break;
            }
        }
        // Final wrap with self-recursive VK
        let wrapped = ir_prover.agg_prove::<GpuEngine>(
            &[current_proofs.into_iter().next().unwrap()],
            DeferralChildVkKind::RecursiveSelf,
            None,
        )?;

        let hook_prover =
            DeferralHookProver::new::<GpuEngine>(ir_prover.get_vk(), root_system_params());
        warn!("proving deferral hook");
        let hook_proof = hook_prover.prove::<GpuEngine>(wrapped, leaf_children)?;
        Ok(hook_proof)
    }

    warn!("aggregating circuit idx 1 (3 leaves)");
    let hook_proof_1 = aggregate_deferral_tree(&def_circuit_vk, idx1_proofs, idx1_io)?;

    warn!("aggregating circuit idx 2 (1 leaf)");
    let hook_proof_2 = aggregate_deferral_tree(&def_circuit_vk, idx2_proofs, idx2_io)?;

    // =========================================================================
    // SECTION 5: VM aggregation path (app -> leaf -> i4l -> ir).
    // =========================================================================
    let leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        Arc::new(app_pk.get_vk()),
        leaf_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving VM leaf aggregation");
    let leaf_vm_proof =
        leaf_prover.agg_prove_no_def::<GpuEngine>(&app_proof.per_segment, ChildVkKind::App)?;

    let i4l_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving VM internal-for-leaf");
    let i4l_vm_proof =
        i4l_prover.agg_prove_no_def::<GpuEngine>(&[leaf_vm_proof], ChildVkKind::Standard)?;

    let ir_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        i4l_prover.get_vk(),
        internal_system_params(),
        true,
        Some(def_hook_commit),
    );
    warn!("proving VM internal-recursive");
    let ir_vm_proof =
        ir_prover.agg_prove_no_def::<GpuEngine>(&[i4l_vm_proof], ChildVkKind::Standard)?;

    // =========================================================================
    // SECTION 6: Deferral path — feed both hook proofs through the VM inner
    // prover with ProofsType::Deferral.
    // =========================================================================
    let deferral_leaf_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        hook_prover_for_commit.get_vk(),
        leaf_system_params(),
        false,
        Some(def_hook_commit),
    );

    let hook1_pvs = read_hook_pvs(&hook_proof_1);
    let hook2_pvs = read_hook_pvs(&hook_proof_2);
    assert_eq!(hook1_pvs.depth, hook2_pvs.depth);
    let zero_leaf = hash_deferral_commit([F::ZERO; DIGEST_SIZE]);
    let idx0_untouched_root =
        poseidon2_compress_with_capacity(hash_deferral_commit(def_vk_commit), zero_leaf).0;
    let zero_depth1_root = poseidon2_compress_with_capacity(zero_leaf, zero_leaf).0;

    warn!("proving deferral-path leaf A (idx0 untouched + idx1)");
    let leaf_def_proof_a = deferral_leaf_prover.agg_prove::<GpuEngine>(
        &[hook_proof_1],
        ChildVkKind::App,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            idx0_untouched_root,
            idx0_untouched_root,
            hook1_pvs.depth,
            true,
        )),
    )?;
    warn!("proving deferral-path leaf B (idx2 + padded zero)");
    let leaf_def_proof_b = deferral_leaf_prover.agg_prove::<GpuEngine>(
        &[hook_proof_2],
        ChildVkKind::App,
        ProofsType::Deferral,
        Some(make_absent_trace_pvs(
            zero_depth1_root,
            zero_depth1_root,
            hook2_pvs.depth,
            false,
        )),
    )?;

    let def_i4l_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        deferral_leaf_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving deferral-path internal-for-leaf");
    let i4l_def_proof = def_i4l_prover.agg_prove::<GpuEngine>(
        &[leaf_def_proof_a, leaf_def_proof_b],
        ChildVkKind::Standard,
        ProofsType::Deferral,
        None,
    )?;

    let def_ir_prover = InnerProver::<MAX_NUM_PROOFS>::new::<GpuEngine>(
        def_i4l_prover.get_vk(),
        internal_system_params(),
        false,
        Some(def_hook_commit),
    );
    warn!("proving deferral-path internal-recursive");
    let ir_def_proof = def_ir_prover.agg_prove::<GpuEngine>(
        &[i4l_def_proof],
        ChildVkKind::Standard,
        ProofsType::Deferral,
        None,
    )?;

    // =========================================================================
    // SECTION 7: Mix VM + deferral internal-recursive proofs.
    // =========================================================================
    warn!("proving mixed-path aggregation");
    let mixed_proof = ir_prover.agg_prove::<GpuEngine>(
        &[ir_vm_proof, ir_def_proof],
        ChildVkKind::RecursiveSelf,
        ProofsType::Mix,
        None,
    )?;

    // =========================================================================
    // SECTION 8: Wrap once with Combined pathway.
    // =========================================================================
    warn!("proving combined-path wrapper");
    let combined_proof = ir_prover.agg_prove::<GpuEngine>(
        &[mixed_proof],
        ChildVkKind::RecursiveSelf,
        ProofsType::Combined,
        None,
    )?;
    let combined_def_depth = read_def_pvs(&combined_proof).depth.as_canonical_u32();
    let start_depth = combined_def_depth as usize;
    let required_commit_count = 1usize << start_depth;
    let mut initial_commits_for_depth = initial_commits.clone();
    let mut final_commits_for_depth = final_commits.clone();
    initial_commits_for_depth.resize(required_commit_count, [F::ZERO; DIGEST_SIZE]);
    final_commits_for_depth.resize(required_commit_count, [F::ZERO; DIGEST_SIZE]);

    let initial_start_node = merkle_root_from_commits(&initial_commits_for_depth);
    let final_start_node = merkle_root_from_commits(&final_commits_for_depth);
    let initial_merkle_proof = generate_set_merkle_proof(
        memory_dimensions,
        &initial_merkle_tree,
        start_depth,
        initial_start_node,
    );
    let final_merkle_proof = generate_set_merkle_proof(
        memory_dimensions,
        &final_merkle_tree,
        start_depth,
        final_start_node,
    );
    let merkle_proofs = DeferralMerkleProofs {
        initial_merkle_proof,
        final_merkle_proof,
    };

    // =========================================================================
    // SECTION 9: Root prover (CPU) and final verification.
    //
    // The root verifier checks def_hook_vk_commit from the deferral path's
    // [cached_commit, vk_pre_hash] tuples for app/leaf/i4l.
    // =========================================================================
    let vm_ir_vk = ir_prover.get_vk();
    let vm_ir_pcs_data = ir_prover.get_self_vk_pcs_data().unwrap();

    let def_hook_dag_commit = deferral_leaf_prover.get_dag_commit(false);
    let def_leaf_dag_commit = def_i4l_prover.get_dag_commit(false);
    let def_i4l_dag_commit = def_ir_prover.get_dag_commit(false);

    let def_hook_vk_commit = poseidon2_hash_slice_with_states(
        &[
            def_hook_dag_commit.cached_commit,
            def_hook_dag_commit.vk_pre_hash,
            def_leaf_dag_commit.cached_commit,
            def_leaf_dag_commit.vk_pre_hash,
            def_i4l_dag_commit.cached_commit,
            def_i4l_dag_commit.vk_pre_hash,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>(),
    )
    .0;

    let root_prover = RootProver::new::<RootEngine>(
        vm_ir_vk.clone(),
        vm_ir_pcs_data.commitment.into(),
        root_system_params(),
        system.memory_config.memory_dimensions(),
        system.num_public_values,
        Some(def_hook_vk_commit.into()),
        None,
    );
    let ctx = root_prover.generate_proving_ctx_with_deferrals(
        combined_proof,
        &user_pvs_proof,
        &merkle_proofs,
    );
    warn!("proving root (CPU)");
    let root_proof = root_prover.root_prove_from_ctx::<RootEngine>(ctx.unwrap())?;

    let root_vk = root_prover.get_vk();
    let engine = RootEngine::new(root_vk.inner.params.clone());
    engine.verify(&root_vk, &root_proof)?;

    Ok(())
}
