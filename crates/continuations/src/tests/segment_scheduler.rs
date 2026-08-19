//! Equivalence between the serial continuation prover and the scheduler-driven one.
//!
//! Scheduling decides *when* a segment's execute and prove run, never *what* they
//! produce. The strongest form of that claim is byte-identity of
//! `ContinuationVmProof`, which this configuration cannot decide: the baseline
//! serial driver does not reproduce its own proof between two runs, because the
//! periphery Poseidon2 trace is built by iterating a concurrent map whose order is
//! not stable (`openvm_circuit::system::poseidon2`). What is decided here is
//! everything downstream of that: the segmentation, the public values, and the
//! validity of every segment proof.

use eyre::Result;
use openvm_circuit::arch::{
    instructions::exe::VmExe, verify_segments, ContinuationVmProof, ContinuationVmProver,
    SegmentSchedulerConfig, VirtualMachine, VmInstance,
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
const LOG_FIB_INPUT: usize = 20;
const EXPECTED_SEGMENTS: usize = 2;
/// A 32 GB card as `nvidia-smi` reports it: driver and CUDA context already
/// counted, so this is the whole device rather than a workload-only remainder.
const DEVICE_GPU_BYTES: u64 = 32_768 << 20;

#[allow(clippy::type_complexity)]
fn multi_segment_instance() -> Result<(
    VmInstance<Engine, Rv64ImBuilder>,
    MultiStarkVerifyingKey<SC>,
    Vec<u8>,
)> {
    let config = test_rv64im_config();
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
fn scheduled_continuations_agree_with_serial() -> Result<()> {
    // Each arm gets its own instance. Proving the same input twice on one instance
    // does not reproduce that instance's own first proof, so sharing one would
    // compare carry-over state rather than the two drivers.
    let (mut serial_instance, vk, input) = multi_segment_instance()?;
    assert!(
        serial_instance.segment_scheduler().is_none(),
        "the serial driver must be the default"
    );
    let serial = serial_instance.prove(vec![input.clone()])?;
    assert_eq!(
        serial.per_segment.len(),
        EXPECTED_SEGMENTS,
        "the workload must keep spanning several segments for this test to mean anything"
    );
    drop(serial_instance);

    let (mut scheduled_instance, _, _) = multi_segment_instance()?;
    scheduled_instance
        .set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(DEVICE_GPU_BYTES)));
    let scheduled = scheduled_instance.prove(vec![input])?;

    assert_eq!(
        serial.per_segment.len(),
        scheduled.per_segment.len(),
        "segment count must not depend on the driver"
    );
    assert_eq!(
        bitcode::serialize(&serial.user_public_values).unwrap(),
        bitcode::serialize(&scheduled.user_public_values).unwrap(),
        "public values must not depend on the driver"
    );
    verify_segments(&scheduled_instance.vm.engine, &vk, &scheduled.per_segment)
        .map_err(|error| eyre::eyre!("scheduled segment proofs must verify: {error}"))?;
    assert!(
        scheduled_instance.max_concurrent_proves() >= 2,
        "the budget admits two resident proves, so two must have run together; saw {}",
        scheduled_instance.max_concurrent_proves()
    );

    // The comparisons above are only evidence if they can fail. Reversing
    // `per_segment` must be caught, which is what pins the *order* of the vector
    // rather than merely the set of proofs it holds.
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
