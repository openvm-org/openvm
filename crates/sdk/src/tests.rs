use std::{borrow::Borrow, slice::from_ref, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_circuit::{
    arch::{hasher::poseidon2::vm_poseidon2_hasher, instructions::DEFERRAL_AS},
    system::memory::merkle::MerkleTree,
};
use openvm_continuations::{circuit::inner::ProofsType, prover::ChildVkKind};
use openvm_deferral_circuit::DeferralFn;
use openvm_stark_backend::{
    p3_field::PrimeField32,
    proof::Proof,
    verifier::{batch_constraints::BatchConstraintError, VerifierError},
    StarkEngine,
};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security, internal_params_with_100_bits_security,
        root_params_with_100_bits_security,
    },
    utils::setup_tracing,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_circuit::extension::{
    get_deferral_state, get_raw_deferral_results, verify_stark_deferral_fn,
};
use openvm_verify_stark_host::{
    error::VerifyStarkError,
    pvs::{DeferralPvs, VerifierBasePvs, VerifierDefPvs, DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID},
    vk::VmStarkVerifyingKey,
    VmStarkProof,
};

use crate::{
    config::{
        AggregationConfig, AggregationSystemParams, AggregationTreeConfig, AppConfig,
        DEFAULT_APP_L_SKIP,
    },
    error::SdkError,
    prover::{
        compute_deferral_merkle_proofs, DeferralProof, DeferralProver, InternalLayerMetadata,
    },
    DeferralInput, Sdk, StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        type RootE = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

#[test]
fn test_sdk_fibonacci() -> Result<()> {
    setup_tracing();
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let sdk = Sdk::riscv32(app_params, agg_params);
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    #[cfg(not(feature = "evm-verify"))]
    {
        let mut evm_prover = sdk.evm_prover(app_exe)?;
        let proof = evm_prover.prove_unwrapped(stdin, &[])?;
        let vk = evm_prover.root_prover.0.get_vk();
        let engine = RootE::new(vk.inner.params.clone());
        engine.verify(&vk, &proof)?;
    }
    #[cfg(feature = "evm-verify")]
    {
        let evm_proof = sdk.prove_evm(app_exe, stdin, &[])?;
        let openvm_verifier = sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof)?;
    }

    Ok(())
}

#[test]
fn test_verify_stark_deferral() -> Result<()> {
    // ---- Step 1: Create a fibonacci proof ----
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let fib_elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let fib_sdk = Sdk::riscv32(app_params.clone(), agg_params.clone());
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;

    let n = 100u64;
    let mut fib_stdin = StdIn::default();
    fib_stdin.write(&n);

    let (fib_proof, fib_baseline) = fib_sdk.prove(fib_exe, fib_stdin, &[])?;

    // ---- Step 2: Build the DeferredVerifyCircuitProver ----
    let fib_agg_prover = fib_sdk.agg_prover();
    let ir_vk = fib_agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = fib_agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();

    let fib_system_config = fib_sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = fib_system_config.memory_config.memory_dimensions();
    let num_user_pvs = fib_system_config.num_public_values;

    let def_circuit_params = internal_params_with_100_bits_security();
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        0,
    );
    let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);

    // ---- Step 3: Create DeferralProver ----
    let hook_params = root_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);

    // ---- Step 4: Create DeferralExtension ----
    let deferral_ext =
        deferral_prover.make_extension(vec![Arc::new(DeferralFn::new(verify_stark_deferral_fn))]);

    // ---- Step 5: Compute deferral state and guest stdin values ----
    let fib_vk = VmStarkVerifyingKey {
        mvk: fib_sdk.agg_vk().as_ref().clone(),
        baseline: fib_baseline,
    };

    // Get the raw results to extract input_commit and output for the guest stdin
    let raw_results = get_raw_deferral_results(&fib_vk, from_ref(&fib_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vm_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();
    let user_public_values = output_raw[64..].to_vec();

    // Build the deferral state for execution
    let deferral_state = get_deferral_state(&fib_vk, from_ref(&fib_proof), 0)?;

    // ---- Step 6: Create verify-stark SDK with deferral ----
    let mut vs_config = openvm_sdk_config::SdkVmConfig::riscv32();
    vs_config.deferral = Some(deferral_ext);
    vs_config.system.config.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;

    let vs_app_config = AppConfig::new(vs_config, app_params);
    let vs_sdk = Sdk::builder()
        .app_config(vs_app_config)
        .agg_params(agg_params)
        .deferral_prover(deferral_prover)
        .build()?;

    // ---- Step 7: Build the verify-stark ELF ----
    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    // ---- Step 8: Set up stdin for the verify-stark guest program ----
    let mut vs_stdin = StdIn::default();
    vs_stdin.write(&app_exe_commit);
    vs_stdin.write(&app_vm_commit);
    vs_stdin.write(&user_public_values);
    vs_stdin.write(&input_commit);
    vs_stdin.deferrals = vec![deferral_state];

    // ---- Step 9: Create DeferralInput from the fibonacci proof ----
    let def_input = DeferralInput::from_inputs(&[fib_proof]);

    // ---- Step 10: Prove and verify ----
    let mut evm_prover = vs_sdk.evm_prover(vs_exe)?;
    let vs_proof = evm_prover.prove_unwrapped(vs_stdin, &[def_input])?;

    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &vs_proof)?;

    Ok(())
}

#[test]
fn test_prove_mixed_requires_stage_aligned_deferral_child() -> Result<()> {
    // ---- Step 1: Create a fibonacci proof ----
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let fib_elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let fib_sdk = Sdk::riscv32(app_params.clone(), agg_params.clone());
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;

    let n = 100u64;
    let mut fib_stdin = StdIn::default();
    fib_stdin.write(&n);

    let (fib_proof, fib_baseline) = fib_sdk.prove(fib_exe, fib_stdin, &[])?;

    // ---- Step 2: Build the DeferredVerifyCircuitProver ----
    let fib_agg_prover = fib_sdk.agg_prover();
    let ir_vk = fib_agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = fib_agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();

    let fib_system_config = fib_sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = fib_system_config.memory_config.memory_dimensions();
    let num_user_pvs = fib_system_config.num_public_values;

    let def_circuit_params = internal_params_with_100_bits_security();
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        0,
    );
    let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);

    // ---- Step 3: Create DeferralProver and extension ----
    let hook_params = root_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);
    let deferral_ext =
        deferral_prover.make_extension(vec![Arc::new(DeferralFn::new(verify_stark_deferral_fn))]);

    // ---- Step 4: Compute deferral state and guest stdin values ----
    let fib_vk = VmStarkVerifyingKey {
        mvk: fib_sdk.agg_vk().as_ref().clone(),
        baseline: fib_baseline,
    };

    let raw_results = get_raw_deferral_results(&fib_vk, from_ref(&fib_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vm_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();
    let user_public_values = output_raw[64..].to_vec();
    let deferral_state = get_deferral_state(&fib_vk, from_ref(&fib_proof), 0)?;

    // ---- Step 5: Build verify-stark SDK with a forced aggregation tree ----
    let mut vs_config = openvm_sdk_config::SdkVmConfig::riscv32();
    vs_config.deferral = Some(deferral_ext);
    vs_config.system.config.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;

    let vs_app_config = AppConfig::new(vs_config, app_params);
    let vs_sdk = Sdk::builder()
        .app_config(vs_app_config)
        .agg_params(agg_params)
        .agg_tree_config(AggregationTreeConfig {
            num_children_leaf: 1,
            num_children_internal: 2,
        })
        .deferral_prover(deferral_prover)
        .build()?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    let mut vs_stdin = StdIn::default();
    vs_stdin.write(&app_exe_commit);
    vs_stdin.write(&app_vm_commit);
    vs_stdin.write(&user_public_values);
    vs_stdin.write(&input_commit);
    vs_stdin.deferrals = vec![deferral_state];

    let def_input = DeferralInput::from_inputs(&[fib_proof]);

    // ---- Step 6: Prove the VM child and deferral child separately ----
    let mut stark_prover = vs_sdk.prover(vs_exe)?;
    let memory_dimensions = stark_prover.app_prover.memory_dimensions();
    let hasher = vm_poseidon2_hasher();
    let initial_memory = &stark_prover
        .app_prover
        .instance()
        .state()
        .as_ref()
        .expect("initial state should exist before proving")
        .memory
        .memory;
    let initial_merkle_tree = MerkleTree::from_memory(initial_memory, &memory_dimensions, &hasher);

    let continuation_proof = stark_prover.app_prover.prove(vs_stdin)?;
    eprintln!(
        "verify-stark continuation segments = {}",
        continuation_proof.per_segment.len()
    );

    let (mut vm_proof, mut metadata) = stark_prover.agg_prover.prove_vm(continuation_proof)?;
    while verifier_base_pvs(&vm_proof.inner)
        .recursion_flag
        .as_canonical_u32()
        < 2
    {
        vm_proof = stark_prover
            .agg_prover
            .wrap_proof(vm_proof, &mut metadata)?;
    }
    let def_path_prover = stark_prover
        .def_prover
        .as_ref()
        .expect("verify-stark repro requires deferral path prover");
    let def_hook_proofs = def_path_prover.deferral_prover.prove(&[def_input])?;
    let def_proof = def_path_prover.agg_prover.prove_def(def_hook_proofs)?;

    let vm_base = verifier_base_pvs(&vm_proof.inner);
    let vm_def = verifier_def_pvs(&vm_proof.inner);
    let def_inner = match def_proof {
        DeferralProof::Present(proof) => proof,
        DeferralProof::Absent(_) => panic!("repro expects a present deferral proof"),
    };
    let def_base = verifier_base_pvs(&def_inner);
    let def_def = verifier_def_pvs(&def_inner);

    eprintln!(
        "vm verifier base = ({}, {}), deferral_flag = {}",
        vm_base.internal_flag.as_canonical_u32(),
        vm_base.recursion_flag.as_canonical_u32(),
        vm_def.deferral_flag.as_canonical_u32()
    );
    eprintln!(
        "def verifier base = ({}, {}), deferral_flag = {}",
        def_base.internal_flag.as_canonical_u32(),
        def_base.recursion_flag.as_canonical_u32(),
        def_def.deferral_flag.as_canonical_u32()
    );

    assert_eq!(vm_base.internal_flag.as_canonical_u32(), 2);
    assert_eq!(vm_base.recursion_flag.as_canonical_u32(), 2);
    assert_eq!(vm_def.deferral_flag.as_canonical_u32(), 0);
    assert_eq!(def_base.internal_flag.as_canonical_u32(), 2);
    assert_eq!(def_base.recursion_flag.as_canonical_u32(), 1);
    assert_eq!(def_def.deferral_flag.as_canonical_u32(), 1);
    assert_ne!(
        vm_base.internal_recursive_vk_commit,
        def_base.internal_recursive_vk_commit,
        "expected the proof-embedded recursive-self VK commit visibility to differ before alignment"
    );

    let final_memory = &stark_prover
        .app_prover
        .instance()
        .state()
        .as_ref()
        .expect("final state should exist after proving")
        .memory
        .memory;
    let final_merkle_tree = MerkleTree::from_memory(final_memory, &memory_dimensions, &hasher);

    // ---- Step 7: Show that the pre-fix beta.2 prove_mixed path returns a proof but fails ----
    let mut mixed_proof = legacy_prove_mixed_unaligned(
        &stark_prover.agg_prover,
        vm_proof.clone(),
        def_inner.clone(),
        &mut copy_metadata(&metadata),
    )?;
    attach_deferral_merkle_proofs(
        &mut mixed_proof,
        memory_dimensions,
        &initial_merkle_tree,
        &final_merkle_tree,
    );

    let mixed_err = Sdk::verify_proof(
        vs_sdk.agg_vk().as_ref().clone(),
        stark_prover.generate_baseline(),
        &mixed_proof,
    )
    .expect_err("unaligned prove_mixed should return a proof whose host verification fails");
    assert_sum_claim_mismatch(mixed_err);

    // ---- Step 8: Manually lift the deferral child until recursion_flag matches ----
    let mut aligned_def_inner = def_inner;
    while verifier_base_pvs(&aligned_def_inner)
        .recursion_flag
        .as_canonical_u32()
        < vm_base.recursion_flag.as_canonical_u32()
    {
        aligned_def_inner = stark_prover
            .agg_prover
            .internal_recursive_prover
            .agg_prove::<E>(
                &[aligned_def_inner],
                ChildVkKind::RecursiveSelf,
                ProofsType::Deferral,
                None,
            )?;
    }

    let aligned_def_base = verifier_base_pvs(&aligned_def_inner);
    eprintln!(
        "aligned def verifier base = ({}, {})",
        aligned_def_base.internal_flag.as_canonical_u32(),
        aligned_def_base.recursion_flag.as_canonical_u32(),
    );
    assert_eq!(aligned_def_base.internal_flag.as_canonical_u32(), 2);
    assert_eq!(
        aligned_def_base.recursion_flag.as_canonical_u32(),
        vm_base.recursion_flag.as_canonical_u32()
    );
    assert_eq!(
        aligned_def_base.internal_recursive_vk_commit,
        vm_base.internal_recursive_vk_commit,
        "expected the aligned deferral child to encode the same recursive-self VK commit as the VM child"
    );

    let mut aligned_mixed_proof = stark_prover.agg_prover.prove_mixed(
        vm_proof,
        DeferralProof::Present(aligned_def_inner),
        &mut copy_metadata(&metadata),
    )?;
    attach_deferral_merkle_proofs(
        &mut aligned_mixed_proof,
        memory_dimensions,
        &initial_merkle_tree,
        &final_merkle_tree,
    );

    Sdk::verify_proof(
        vs_sdk.agg_vk().as_ref().clone(),
        stark_prover.generate_baseline(),
        &aligned_mixed_proof,
    )
    .expect("aligned prove_mixed output should host-verify");

    Ok(())
}

#[test]
fn test_deferrals_enabled_without_usage() -> Result<()> {
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    // ---- Step 1: Create dummy DeferralProver ----
    let rv32_sdk = Sdk::riscv32(app_params.clone(), agg_params.clone());
    let ir_prover = &rv32_sdk.agg_prover().internal_recursive_prover;
    let ir_vk = ir_prover.get_vk();
    let ir_pcs_data = ir_prover.get_self_vk_pcs_data().unwrap();

    let system_config = rv32_sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;

    let def_circuit_params = internal_params_with_100_bits_security();
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        0,
    );
    let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);

    let hook_params = root_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);

    // ---- Step 2: Enable deferrals in SDK and prove ----
    let sdk = Sdk::builder()
        .app_config(AppConfig::riscv32(app_params))
        .agg_params(agg_params.clone())
        .deferral_prover(deferral_prover)
        .build()?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let mut evm_prover = sdk.evm_prover(app_exe)?;
    let proof = evm_prover.prove_unwrapped(stdin, &[])?;

    // ---- Step 3: Verify the final result ----
    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &proof)?;

    Ok(())
}

fn verifier_base_pvs(proof: &Proof<crate::SC>) -> VerifierBasePvs<crate::F> {
    let slice = proof.public_values[VERIFIER_PVS_AIR_ID].as_slice();
    let pvs: &VerifierBasePvs<crate::F> = slice[..VerifierBasePvs::<u8>::width()].borrow();
    *pvs
}

fn verifier_def_pvs(proof: &Proof<crate::SC>) -> VerifierDefPvs<crate::F> {
    let slice = proof.public_values[VERIFIER_PVS_AIR_ID].as_slice();
    let (_, def_slice) = slice.split_at(VerifierBasePvs::<u8>::width());
    let pvs: &VerifierDefPvs<crate::F> = def_slice.borrow();
    *pvs
}

fn copy_metadata(metadata: &InternalLayerMetadata) -> InternalLayerMetadata {
    InternalLayerMetadata {
        internal_recursive_layer: metadata.internal_recursive_layer,
        internal_node_idx: metadata.internal_node_idx,
        proofs_type: metadata.proofs_type,
    }
}

fn attach_deferral_merkle_proofs(
    proof: &mut VmStarkProof,
    memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
    initial_merkle_tree: &MerkleTree<crate::F, 8>,
    final_merkle_tree: &MerkleTree<crate::F, 8>,
) {
    let def_pvs: &DeferralPvs<crate::F> = proof.inner.public_values[DEF_PVS_AIR_ID]
        .as_slice()
        .borrow();
    let depth = def_pvs.depth.as_canonical_u32() as usize;
    proof.deferral_merkle_proofs = Some(compute_deferral_merkle_proofs(
        memory_dimensions,
        initial_merkle_tree,
        final_merkle_tree,
        depth,
    ));
}

fn legacy_prove_mixed_unaligned(
    agg_prover: &crate::prover::AggProver,
    mut vm_proof: VmStarkProof,
    def_inner: Proof<crate::SC>,
    metadata: &mut InternalLayerMetadata,
) -> Result<VmStarkProof> {
    vm_proof.inner = agg_prover.internal_recursive_prover.agg_prove::<E>(
        &[vm_proof.inner, def_inner],
        ChildVkKind::RecursiveSelf,
        ProofsType::Mix,
        None,
    )?;
    metadata.internal_recursive_layer += 1;
    metadata.internal_node_idx += 1;
    metadata.proofs_type = ProofsType::Combined;
    Ok(vm_proof)
}

fn assert_sum_claim_mismatch(err: SdkError) {
    match err {
        SdkError::VerifyStark(VerifyStarkError::StarkVerificationFailure(
            VerifierError::BatchConstraintError(BatchConstraintError::SumClaimMismatch { .. }),
        )) => {}
        other => panic!(
            "expected unaligned mixed proof to fail with BatchConstraintError::SumClaimMismatch, got {other:?}"
        ),
    }
}

/// Cell-count profiling test for the static verifier circuit using a production root proof.
///
/// Root verifier params match `pipeline_cell_count_profiling` in static-verifier crate.
/// The root proof is generated from a full SDK aggregation pipeline and cached to disk.
///
/// Run with:
/// ```sh
/// OPENVM_CACHE_DIR=cache OPENVM_PROFILE_DIR=profile \
///   cargo nextest run --cargo-profile=fast -p openvm-sdk --features cuda,cell-profiling \
///   -- sdk_static_verifier_cell_profiling
/// ```
#[cfg(feature = "cell-profiling")]
#[test]
fn sdk_static_verifier_cell_profiling() -> Result<()> {
    use std::path::Path;

    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use openvm::platform::memory::MEM_SIZE;
    use openvm_continuations::{CommitBytes, RootSC};
    use openvm_stark_backend::{
        codec::{Decode, Encode},
        proof::Proof,
    };
    use openvm_static_verifier::{
        compute_dag_onion_commit,
        field::baby_bear::{BabyBearChip, BabyBearExtChip},
        log_heights_per_air_from_proof, StaticVerifierCircuit,
    };

    use crate::{
        config::{AggregationSystemParams, DEFAULT_APP_L_SKIP},
        keygen::dummy::compute_root_proof_heights,
        prover::{EvmProver, RootProver},
        Sdk, StdIn,
    };

    // Root verifier params matching pipeline_cell_count_profiling in static-verifier
    let root_params = root_params_with_100_bits_security();
    let cache_dir = std::env::var("OPENVM_CACHE_DIR").unwrap_or_else(|_| "cache".to_string());
    std::fs::create_dir_all(&cache_dir)?;

    let proof_path = format!("{cache_dir}/sdk_root_proof.bin");
    let vk_path = format!("{cache_dir}/sdk_root_vk.bin");
    let commit_path = format!("{cache_dir}/sdk_onion_commit.bin");

    let (root_vk, root_proof, onion_commit) =
        if Path::new(&proof_path).exists() && Path::new(&vk_path).exists() {
            eprintln!("Loading cached root proof from {cache_dir}/");
            let proof_bytes = std::fs::read(&proof_path)?;
            let root_proof = Proof::<RootSC>::decode_from_bytes(&proof_bytes)?;

            let vk_bytes = std::fs::read(&vk_path)?;
            let root_vk = bitcode::deserialize(&vk_bytes)
                .map_err(|e| eyre::eyre!("failed to deserialize root VK: {e}"))?;

            let commit_bytes: [u8; 32] = std::fs::read(&commit_path)?
                .try_into()
                .map_err(|_| eyre::eyre!("invalid commit file"))?;
            let onion_commit = CommitBytes::new(commit_bytes).into();

            (root_vk, root_proof, onion_commit)
        } else {
            eprintln!("Generating root proof via SDK pipeline (this takes a while)...");
            let n_stack = 19;
            let app_params = openvm_stark_sdk::config::app_params_with_100_bits_security(
                DEFAULT_APP_L_SKIP + n_stack,
            );
            let agg_params = AggregationSystemParams::default();

            let elf = Elf::decode(
                include_bytes!("../programs/examples/fibonacci.elf"),
                MEM_SIZE as u32,
            )?;
            let sdk = Sdk::riscv32(app_params, agg_params);
            let app_exe = sdk.convert_to_exe(elf)?;

            // Compute trace heights for root prover with profiling params
            let system_config = sdk.app_config().app_vm_config.as_ref();
            let agg_prover = sdk.agg_prover();
            let (trace_heights, root_pk) = compute_root_proof_heights::<E, _>(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                agg_prover.clone(),
                root_params.clone(),
                None,
            )?;

            let ir_vk = agg_prover.internal_recursive_prover.get_vk();
            let ir_pcs_data = agg_prover
                .internal_recursive_prover
                .get_self_vk_pcs_data()
                .unwrap();
            let vk_commit: CommitBytes = ir_pcs_data.commitment.into();
            let onion_commit = compute_dag_onion_commit(&ir_vk);

            let memory_dimensions = system_config.memory_config.memory_dimensions();
            let num_user_pvs = system_config.num_public_values;

            let root_prover = Arc::new(RootProver::from_pk(
                ir_vk,
                vk_commit,
                root_pk,
                memory_dimensions,
                num_user_pvs,
                None,
                Some(trace_heights),
            ));

            let mut evm_prover = EvmProver::<E, _>::new(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                app_exe,
                agg_prover,
                None,
                root_prover.clone(),
                None,
            )?;

            let n = 100u64;
            let mut stdin = StdIn::default();
            stdin.write(&n);

            let root_proof = evm_prover.prove_unwrapped(stdin, &[])?;
            let root_vk_arc = root_prover.0.get_vk();
            let root_vk = root_vk_arc.as_ref().clone();

            // Verify the root proof
            let engine = RootE::new(root_vk.inner.params.clone());
            engine.verify(&root_vk, &root_proof)?;

            // Cache to disk
            eprintln!("Caching root proof to {cache_dir}/");
            std::fs::write(&proof_path, root_proof.encode_to_vec()?)?;
            std::fs::write(
                &vk_path,
                bitcode::serialize(&root_vk)
                    .map_err(|e| eyre::eyre!("failed to serialize root VK: {e}"))?,
            )?;
            std::fs::write(&commit_path, CommitBytes::from(onion_commit).as_slice())?;

            (root_vk, root_proof, onion_commit)
        };

    // Run static verifier cell profiling
    eprintln!("Running static verifier cell profiling...");
    let log_heights = log_heights_per_air_from_proof(&root_proof);

    let circuit = StaticVerifierCircuit::try_new(root_vk, onion_commit, &log_heights)
        .expect("Failed to construct StaticVerifierCircuit");

    let profile_dir = std::env::var("OPENVM_PROFILE_DIR").unwrap_or_else(|_| "profile".to_string());
    std::env::set_var("OPENVM_PROFILE_DIR", &profile_dir);

    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(22)
        .use_lookup_bits(21)
        .use_instance_columns(0);
    let range = builder.range_chip();
    let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
    let ctx = builder.main(0);

    let initial_cells = ctx.advice.len();
    circuit.populate_verify_stark_constraints(ctx, &ext_chip, &root_proof);
    let final_cells = ctx.advice.len();
    eprintln!(
        "Static verifier cell count: {} (delta: {})",
        final_cells,
        final_cells - initial_cells
    );
    assert!(
        final_cells > initial_cells,
        "expected advice cells to increase"
    );

    Ok(())
}
