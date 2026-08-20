//! Equivalence between the serial continuation prover and the scheduler-driven one.
//!
//! Scheduling decides *when* a segment's execute and prove run, never *what* they
//! produce. Proving that at the byte level needs a prover that reproduces itself,
//! and by default this one does not: the periphery Poseidon2 trace is built by
//! iterating a concurrent map, so its row order follows insertion history rather
//! than the records. The two tests here split that apart.
//!
//! - Under [`set_deterministic_tracegen`] the serial driver reproduces itself, so the scheduled
//!   driver can be held to byte-identity. That is the scheduling question, isolated.
//! - Under the default ordering the proofs may differ byte for byte, so what must match is their
//!   meaning: the executable commitment, the final memory root, the public values, and every
//!   segment's public values, with both sides verifying.

use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, verify_segments,
        ContinuationVmProof, ContinuationVmProver, SegmentSchedulerConfig, VirtualMachine,
        VmInstance,
    },
    system::poseidon2::set_deterministic_tracegen,
};
use openvm_riscv_circuit::Rv64ImBuilder;
use openvm_riscv_transpiler::{
    Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, StarkEngine};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};

use super::{app_system_params, test_rv64im_config, Engine};
use crate::SC;

/// Fibonacci input that splits into more than one segment under
/// [`test_rv64im_config`] while staying inside the app parameters' stacked height,
/// so the graph has a real execute chain and an observable `per_segment` order.
const LOG_FIB_INPUT: usize = 12;
const EXPECTED_SEGMENTS: usize = 2;
/// Segmentation ceiling low enough that a small input still splits, so the graph
/// is real without paying for a large proof.
const SEGMENTATION_MAX_MEMORY: usize = 1 << 27;
/// A 32 GB card as `nvidia-smi` reports it: driver and CUDA context already
/// counted, so this is the whole device rather than a workload-only remainder.
const DEVICE_GPU_BYTES: u64 = 32_768 << 20;

#[allow(clippy::type_complexity)]
fn multi_segment_instance() -> Result<(
    VmInstance<Engine, Rv64ImBuilder>,
    MultiStarkVerifyingKey<SC>,
    Vec<u8>,
)> {
    let mut config = test_rv64im_config();
    config.rv64i.system.segmentation_max_memory = SEGMENTATION_MAX_MEMORY;
    let elf = Elf::decode(
        include_bytes!("../../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv64ITranspilerExtension)
            .with_extension(Rv64MTranspilerExtension)
            .with_extension(Rv64IoTranspilerExtension),
    )?;
    let input = (1u64 << LOG_FIB_INPUT).to_le_bytes().to_vec();

    let engine = Engine::new(app_system_params());
    let (vm, pk) = VirtualMachine::new_with_keygen(engine, Rv64ImBuilder, config)?;
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    Ok((instance, vk, input))
}

fn encode(proof: &ContinuationVmProof<SC>) -> Vec<u8> {
    bitcode::serialize(proof).expect("continuation proof is serializable")
}

#[test]
fn scheduler_admits_two_resident_proves_on_a_32gb_device() {
    let config = SegmentSchedulerConfig::for_device(DEVICE_GPU_BYTES);
    let two = 2 * config.prove.gpu_bytes;
    let three = 3 * config.prove.gpu_bytes;
    assert!(
        two <= config.budget.gpu_bytes,
        "two resident proves must fit a 32 GB device: {two} > {}",
        config.budget.gpu_bytes
    );
    assert!(
        three > config.budget.gpu_bytes,
        "a third resident prove must not fit"
    );
}

#[test]
fn scheduled_continuations_are_byte_identical_under_deterministic_tracegen() -> Result<()> {
    // Two conditions are needed before proofs can be compared as bytes at all, and
    // neither is about scheduling. Trace rows must be ordered by their inputs, and
    // the prover must run on a single worker: with several, the proof this backend
    // emits for one fixed proving key and proving context still varies between
    // runs, below the trace commitment. Both arms run under both conditions, so
    // what remains between them is only when the work was scheduled.
    // The test runs in its own process and nothing has touched the pool yet, so
    // this is read when rayon builds it lazily. Were it already built with more
    // workers, the assertions below would fail rather than quietly pass.
    std::env::set_var("RAYON_NUM_THREADS", "1");
    set_deterministic_tracegen(true);

    // Byte-identity is only a meaningful question once the baseline reproduces
    // itself, so establish that first, on the same driver the scheduler replaces.
    let (mut first_instance, _, input) = multi_segment_instance()?;
    let first = first_instance.prove(vec![input.clone()])?;
    assert_eq!(
        first.per_segment.len(),
        EXPECTED_SEGMENTS,
        "the workload must keep spanning several segments for this test to mean anything"
    );
    drop(first_instance);

    let (mut second_instance, _, _) = multi_segment_instance()?;
    let second = second_instance.prove(vec![input.clone()])?;
    assert_eq!(
        encode(&first),
        encode(&second),
        "ordering tracegen deterministically must make the serial driver reproduce itself"
    );
    drop(second_instance);

    let (mut scheduled_instance, _, _) = multi_segment_instance()?;
    scheduled_instance
        .set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(DEVICE_GPU_BYTES)));
    let scheduled = scheduled_instance.prove(vec![input])?;

    for (idx, (want, got)) in first
        .per_segment
        .iter()
        .zip(scheduled.per_segment.iter())
        .enumerate()
    {
        assert_eq!(
            bitcode::serialize(want).unwrap(),
            bitcode::serialize(got).unwrap(),
            "segment {idx} proof differs"
        );
    }
    assert_eq!(
        encode(&first),
        encode(&scheduled),
        "whole continuation proof differs"
    );
    assert!(
        scheduled_instance.max_concurrent_proves() >= 2,
        "the budget admits two resident proves, so two must have run together; saw {}",
        scheduled_instance.max_concurrent_proves()
    );

    // The equality above is only evidence if it can fail. Reversing `per_segment`
    // must be caught, which is what pins the *order* of the vector rather than
    // merely the set of proofs it holds.
    let reordered = ContinuationVmProof {
        per_segment: scheduled.per_segment.iter().rev().cloned().collect(),
        user_public_values: scheduled.user_public_values.clone(),
    };
    assert_ne!(
        encode(&scheduled),
        encode(&reordered),
        "the byte comparison must discriminate segment order"
    );
    Ok(())
}

#[test]
fn scheduled_continuations_mean_the_same_in_default_mode() -> Result<()> {
    // Each arm gets its own instance. Proving the same input twice on one instance
    // does not reproduce that instance's own first proof, so sharing one would
    // compare carry-over state rather than the two drivers.
    let (mut serial_instance, vk, input) = multi_segment_instance()?;
    assert!(
        serial_instance.segment_scheduler().is_none(),
        "the serial driver must be the default"
    );
    let serial = serial_instance.prove(vec![input.clone()])?;
    assert_eq!(serial.per_segment.len(), EXPECTED_SEGMENTS);
    let serial_payload = verify_segments(&serial_instance.vm.engine, &vk, &serial.per_segment)
        .map_err(|error| eyre::eyre!("serial segment proofs must verify: {error}"))?;
    let serial_instance_boundaries = serial_instance.segment_boundaries().to_vec();
    drop(serial_instance);

    let (mut scheduled_instance, _, _) = multi_segment_instance()?;
    scheduled_instance
        .set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(DEVICE_GPU_BYTES)));
    let scheduled = scheduled_instance.prove(vec![input])?;
    let scheduled_payload =
        verify_segments(&scheduled_instance.vm.engine, &vk, &scheduled.per_segment)
            .map_err(|error| eyre::eyre!("scheduled segment proofs must verify: {error}"))?;

    assert_eq!(
        serial.per_segment.len(),
        scheduled.per_segment.len(),
        "segment count must not depend on the driver"
    );
    assert_eq!(
        serial_payload.exe_commit, scheduled_payload.exe_commit,
        "executable commitment must not depend on the driver"
    );
    assert_eq!(
        serial_payload.final_memory_root, scheduled_payload.final_memory_root,
        "final memory root must not depend on the driver"
    );
    assert_eq!(
        serial.user_public_values.public_values, scheduled.user_public_values.public_values,
        "user public values must not depend on the driver"
    );
    // Each arm's Merkle proof is checked against the root that arm's own segment
    // proofs established, so a scheduled run cannot pass by borrowing the serial
    // run's memory root.
    let memory_dimensions = test_rv64im_config()
        .rv64i
        .system
        .memory_config
        .memory_dimensions();
    let hasher = vm_poseidon2_hasher();
    serial
        .user_public_values
        .verify(&hasher, memory_dimensions, serial_payload.final_memory_root)
        .map_err(|error| eyre::eyre!("serial public values proof must verify: {error}"))?;
    scheduled
        .user_public_values
        .verify(
            &hasher,
            memory_dimensions,
            scheduled_payload.final_memory_root,
        )
        .map_err(|error| eyre::eyre!("scheduled public values proof must verify: {error}"))?;
    for (idx, (want, got)) in serial
        .per_segment
        .iter()
        .zip(scheduled.per_segment.iter())
        .enumerate()
    {
        assert_eq!(
            want.public_values, got.public_values,
            "segment {idx} public values differ"
        );
    }
    assert!(
        scheduled_instance.max_concurrent_proves() >= 2,
        "the budget admits two resident proves, so two must have run together; saw {}",
        scheduled_instance.max_concurrent_proves()
    );

    // The input envelope: what each arm actually handed the prover, per segment.
    //
    // `common_main_commit` is the prover's own commitment to all of a segment's
    // main traces, so comparing it commits to private trace contents without
    // reading a device buffer or synchronizing a stream — which is why this can be
    // an assertion rather than an instrumented comparison. `trace_vdata` adds the
    // per-AIR shapes and cached commitments, and the boundaries pin the segment
    // this trace was supposed to cover, which public values alone do not: the
    // connector exposes program counters and termination but neither instruction
    // count nor timestamps.
    assert_eq!(
        serial_instance_boundaries,
        scheduled_instance.segment_boundaries(),
        "segment boundaries must not depend on the driver"
    );
    for (idx, (want, got)) in serial
        .per_segment
        .iter()
        .zip(scheduled.per_segment.iter())
        .enumerate()
    {
        assert_eq!(
            bitcode::serialize(&want.common_main_commit).unwrap(),
            bitcode::serialize(&got.common_main_commit).unwrap(),
            "segment {idx} committed to different main traces"
        );
        assert_eq!(
            bitcode::serialize(&want.trace_vdata).unwrap(),
            bitcode::serialize(&got.trace_vdata).unwrap(),
            "segment {idx} has different trace shapes or cached commitments"
        );
    }
    Ok(())
}
