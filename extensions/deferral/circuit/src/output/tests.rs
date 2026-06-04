use std::sync::{atomic::Ordering, Arc};

use openvm_circuit::arch::{
    deferral::{DeferralResult, DeferralState},
    testing::{
        memory::{gen_pointer, gen_register_pointer},
        TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder,
    },
    ExecutionError, Executor, MemoryConfig, Postflight, MEMORY_BLOCK_BYTES,
};
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng, RngCore};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    super::DeferralOutputChipGpu,
    crate::{count::DeferralCircuitCountChipGpu, poseidon2::DeferralPoseidon2ChipGpu},
    openvm_circuit::arch::testing::{
        default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_cuda_common::d_buffer::DeviceBuffer,
};

use super::{DeferralOutputAir, DeferralOutputChip, DeferralOutputExecutor, DeferralOutputFiller};
use crate::{
    count::{DeferralCircuitCountAir, DeferralCircuitCountBus, DeferralCircuitCountChip},
    generate_deferral_results,
    poseidon2::{
        deferral_poseidon2_air, deferral_poseidon2_chip, DeferralPoseidon2Air,
        DeferralPoseidon2Bus, DeferralPoseidon2Chip,
    },
    utils::{combine_output, COMMIT_NUM_BYTES, OUTPUT_TOTAL_BYTES, SPONGE_BYTES_PER_ROW},
    RawDeferralResult,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 1024;
const NUM_DEFERRALS: usize = 4;
const DEFERRAL_COUNT_BUS: BusIndex = 20;
const DEFERRAL_POSEIDON2_BUS: BusIndex = 21;

type Harness = TestChipHarness<F, DeferralOutputExecutor, DeferralOutputAir, DeferralOutputChip<F>>;
type CountPeriphery = (DeferralCircuitCountAir, Arc<DeferralCircuitCountChip>);
type Poseidon2Periphery = (DeferralPoseidon2Air<F>, Arc<DeferralPoseidon2Chip<F>>);

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    DeferralOutputExecutor,
    DeferralOutputAir,
    DeferralOutputChipGpu,
    DeferralOutputChip<F>,
>;
#[cfg(all(feature = "cuda", feature = "rvr"))]
type CudaCountPeriphery = (DeferralCircuitCountAir, DeferralCircuitCountChipGpu);
#[cfg(all(feature = "cuda", feature = "rvr"))]
type CudaPoseidon2Periphery = (DeferralPoseidon2Air<F>, DeferralPoseidon2ChipGpu);

struct CpuHarnessBundle {
    harness: Harness,
    count: CountPeriphery,
    poseidon2: Poseidon2Periphery,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct CudaHarnessBundle {
    harness: GpuHarness,
    count: CudaCountPeriphery,
    poseidon2: CudaPoseidon2Periphery,
}

fn test_memory_config() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 20;
    config
}

fn init_streams(tester: &mut impl TestBuilder<F>, num_deferrals: usize) {
    tester.streams_mut().deferrals = vec![DeferralState::new(vec![]); num_deferrals];
}

fn write_output_key(
    tester: &mut impl TestBuilder<F>,
    input_ptr: usize,
    output_key: [u8; OUTPUT_TOTAL_BYTES],
) {
    for (chunk_idx, chunk) in output_key.chunks_exact(MEMORY_BLOCK_BYTES).enumerate() {
        let chunk: [u8; MEMORY_BLOCK_BYTES] = chunk.try_into().unwrap();
        tester.write_bytes(
            RV64_MEMORY_AS as usize,
            input_ptr + chunk_idx * MEMORY_BLOCK_BYTES,
            chunk.map(F::from_u8),
        );
    }
}

fn make_result(
    deferral_idx: usize,
    input_commit: [u8; COMMIT_NUM_BYTES],
    output_raw: Vec<u8>,
) -> DeferralResult {
    let hasher = deferral_poseidon2_chip::<F>();
    generate_deferral_results(
        vec![RawDeferralResult::new(input_commit.to_vec(), output_raw)],
        deferral_idx as u32,
        &hasher,
    )
    .into_iter()
    .next()
    .unwrap()
}

fn set_and_execute_output<E, T>(
    tester: &mut T,
    executor: &mut E,
    preflight: &mut TestPreflight<F>,
    rng: &mut StdRng,
    num_deferrals: usize,
) -> Instruction<F>
where
    E: Executor<F> + Clone,
    T: TestBuilder<F>,
{
    let rd = gen_register_pointer(rng, MEMORY_BLOCK_BYTES);
    let mut rs = gen_register_pointer(rng, MEMORY_BLOCK_BYTES);
    while rs == rd {
        rs = gen_register_pointer(rng, MEMORY_BLOCK_BYTES);
    }
    let output_ptr = gen_pointer(rng, MEMORY_BLOCK_BYTES);
    let input_ptr = gen_pointer(rng, MEMORY_BLOCK_BYTES);
    let deferral_idx = rng.random_range(0..num_deferrals);

    let mut input_commit = [0u8; COMMIT_NUM_BYTES];
    rng.fill_bytes(&mut input_commit);
    let output_len = rng.random_range(0..=4) * SPONGE_BYTES_PER_ROW;
    let mut output_raw = vec![0u8; output_len];
    rng.fill_bytes(&mut output_raw);
    let result = make_result(deferral_idx, input_commit, output_raw);

    let state = &mut tester.streams_mut().deferrals[deferral_idx];
    state.store_input(result.input.clone(), vec![]);
    state.store_output(
        &result.input,
        result.output_commit.clone(),
        result.output_raw.clone(),
    );

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rd,
        (output_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rs,
        (input_ptr as u64).to_le_bytes().map(F::from_u8),
    );

    let output_commit: [u8; COMMIT_NUM_BYTES] = result.output_commit.try_into().unwrap();
    let output_key = combine_output(
        output_commit,
        (result.output_raw.len() as u64).to_le_bytes(),
    );
    write_output_key(tester, input_ptr, output_key);

    let instruction = Instruction::from_usize(
        DeferralOpcode::OUTPUT.global_opcode(),
        [
            rd,
            rs,
            deferral_idx,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    tester.execute(executor, preflight, &instruction);
    instruction
}

fn create_cpu_harness(tester: &VmChipTestBuilder<F>, num_deferrals: usize) -> CpuHarnessBundle {
    let range_checker = tester.range_checker();
    let range_bus = range_checker.bus();
    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

    let air = DeferralOutputAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        count_bus,
        poseidon2_bus,
        range_bus,
        tester.address_bits(),
    );
    let executor = DeferralOutputExecutor::new();
    let chip = DeferralOutputChip::new(
        DeferralOutputFiller::new(
            count_chip.clone(),
            poseidon2_chip.clone(),
            range_checker,
            tester.address_bits(),
        ),
        tester.memory_helper(),
    );

    let harness = Harness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        super::generate_trace_from_postflight,
    )
    .with_rows_used(|trace| {
        trace
            .values
            .chunks_exact(trace.width)
            .take_while(|row| row[0] != F::ZERO)
            .count()
    });
    CpuHarnessBundle {
        harness,
        count: (
            DeferralCircuitCountAir::new(count_bus, num_deferrals),
            count_chip,
        ),
        poseidon2: (deferral_poseidon2_air(poseidon2_bus.0), poseidon2_chip),
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[allow(clippy::type_complexity)]
fn create_cuda_harness(tester: &GpuChipTestBuilder, num_deferrals: usize) -> CudaHarnessBundle {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let range_bus = dummy_range_checker.bus();
    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip_cpu = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip_cpu = Arc::new(deferral_poseidon2_chip());

    let air = DeferralOutputAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        count_bus,
        poseidon2_bus,
        range_bus,
        tester.address_bits(),
    );
    let executor = DeferralOutputExecutor::new();
    let cpu_chip = DeferralOutputChip::new(
        DeferralOutputFiller::new(
            count_chip_cpu,
            poseidon2_chip_cpu,
            dummy_range_checker,
            tester.address_bits(),
        ),
        tester.dummy_memory_helper(),
    );

    let device_ctx = tester.range_checker().device_ctx.clone();
    let count = Arc::new(DeviceBuffer::<u32>::with_capacity_on(
        num_deferrals,
        &device_ctx,
    ));
    count.fill_zero_on(&device_ctx).unwrap();
    let poseidon2_chip_gpu = DeferralPoseidon2ChipGpu::new(1, device_ctx.clone());
    let gpu_chip = DeferralOutputChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
        count.clone(),
        num_deferrals,
        poseidon2_chip_gpu.shared_buffer(),
    );

    let harness = GpuHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
        .with_trace_generators(
            super::generate_trace_from_postflight,
            |chip, program, transcript, plan| {
                chip.generate_proving_ctx_from_postflight(
                    program,
                    transcript,
                    plan,
                    MAX_INS_CAPACITY,
                )
            },
        )
        .with_rows_used(|trace| {
            trace
                .values
                .chunks_exact(trace.width)
                .take_while(|row| row[0] != F::ZERO)
                .count()
        });
    CudaHarnessBundle {
        harness,
        count: (
            DeferralCircuitCountAir::new(count_bus, num_deferrals),
            DeferralCircuitCountChipGpu::new(count, num_deferrals, device_ctx),
        ),
        poseidon2: (deferral_poseidon2_air(poseidon2_bus.0), poseidon2_chip_gpu),
    }
}

#[test]
#[should_panic(expected = "deferral output length must be a multiple of SPONGE_BYTES_PER_ROW")]
fn output_raw_len_must_be_sponge_row_aligned() {
    make_result(0, [0; COMMIT_NUM_BYTES], vec![1]);
}

#[test]
fn rand_deferral_output_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::<F>::from_config(test_memory_config());
    let CpuHarnessBundle {
        mut harness,
        count,
        poseidon2,
    } = create_cpu_harness(&tester, NUM_DEFERRALS);

    init_streams(&mut tester, NUM_DEFERRALS);
    let num_ops = 25;
    for _ in 0..num_ops {
        set_and_execute_output(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            NUM_DEFERRALS,
        );
    }

    tester
        .build()
        .load(harness)
        .load_periphery(count)
        .load_periphery(poseidon2)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[test]
fn postflight_output_trace_rejects_truncated_history_without_mutating_periphery() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::<F>::from_config(test_memory_config());
    let CpuHarnessBundle {
        mut harness,
        count,
        poseidon2,
        ..
    } = create_cpu_harness(&tester, NUM_DEFERRALS);
    init_streams(&mut tester, NUM_DEFERRALS);
    let instruction = set_and_execute_output(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        NUM_DEFERRALS,
    );
    let from_pc = tester.last_from_pc().as_canonical_u32();
    let sentinel = instruction.clone();
    let program = Program::new_without_debug_infos(&[instruction, sentinel], from_pc);
    let history = &mut harness.preflight.executions[0].history;
    let memory_config = test_memory_config();
    let postflight = Postflight::new(&program, history, &memory_config, None).unwrap();
    let actual = super::generate_trace_from_postflight(&harness.chip, &postflight).unwrap();
    assert!(!actual.values.is_empty());
    drop(postflight);

    history
        .memory
        .accesses
        .pop()
        .expect("OUTPUT has timed memory events");
    let postflight = Postflight::new(&program, history, &memory_config, None).unwrap();
    let counts_before = count
        .1
        .count
        .iter()
        .map(|count| count.load(Ordering::Relaxed))
        .collect::<Vec<_>>();
    let poseidon_records_before = poseidon2.1.records.len();
    let error = super::generate_trace_from_postflight(&harness.chip, &postflight)
        .expect_err("truncated OUTPUT history must be rejected");
    assert!(
        error.to_string().contains("too few memory events")
            || error.to_string().contains("ended at timestamp")
    );
    assert_eq!(
        counts_before,
        count
            .1
            .count
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .collect::<Vec<_>>()
    );
    assert_eq!(poseidon_records_before, poseidon2.1.records.len());
}

#[test]
fn deferral_output_rejects_invalid_deferral_index() {
    let error = super::checked_deferral_index(17, NUM_DEFERRALS, NUM_DEFERRALS as u32)
        .expect_err("invalid deferral index must return an execution error");

    assert!(matches!(
        error,
        ExecutionError::Fail {
            pc: 17,
            msg: "deferral index is out of bounds"
        }
    ));
}

/// Regression test that a multi-row OUTPUT section initializes every constrained column.
#[test]
fn deferral_output_multi_row_trace_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::<F>::from_config(test_memory_config());
    let CpuHarnessBundle {
        mut harness,
        count,
        poseidon2,
    } = create_cpu_harness(&tester, NUM_DEFERRALS);

    init_streams(&mut tester, NUM_DEFERRALS);

    let rd = gen_register_pointer(&mut rng, MEMORY_BLOCK_BYTES);
    let rs = gen_register_pointer(&mut rng, MEMORY_BLOCK_BYTES);
    let output_ptr = gen_pointer(&mut rng, MEMORY_BLOCK_BYTES);
    let input_ptr = gen_pointer(&mut rng, MEMORY_BLOCK_BYTES);
    let deferral_idx = 0;

    // Two sponge blocks produce three rows, including the initialization row.
    let output_len = 2 * SPONGE_BYTES_PER_ROW;
    let mut input_commit = [0u8; COMMIT_NUM_BYTES];
    rng.fill_bytes(&mut input_commit);
    let mut output_raw = vec![0u8; output_len];
    rng.fill_bytes(&mut output_raw);
    let result = make_result(deferral_idx, input_commit, output_raw);

    let state = &mut tester.streams_mut().deferrals[deferral_idx];
    state.store_input(result.input.clone(), vec![]);
    state.store_output(
        &result.input,
        result.output_commit.clone(),
        result.output_raw.clone(),
    );

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rd,
        (output_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rs,
        (input_ptr as u64).to_le_bytes().map(F::from_u8),
    );

    let output_commit: [u8; COMMIT_NUM_BYTES] = result.output_commit.try_into().unwrap();
    let output_key = combine_output(
        output_commit,
        (result.output_raw.len() as u64).to_le_bytes(),
    );
    write_output_key(&mut tester, input_ptr, output_key);

    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &Instruction::from_usize(
            DeferralOpcode::OUTPUT.global_opcode(),
            [
                rd,
                rs,
                deferral_idx,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
    );

    tester
        .build()
        .load(harness)
        .load_periphery(count)
        .load_periphery(poseidon2)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_rand_deferral_output_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::new(
        test_memory_config(),
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    );
    let CudaHarnessBundle {
        mut harness,
        count,
        poseidon2,
    } = create_cuda_harness(&tester, NUM_DEFERRALS);

    init_streams(&mut tester, NUM_DEFERRALS);
    let num_ops = 40;
    for _ in 0..num_ops {
        set_and_execute_output(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            NUM_DEFERRALS,
        );
    }

    let mut tester = tester.build().load_gpu_harness(harness);
    let count_ctx = count
        .1
        .generate_proving_ctx_direct(MAX_INS_CAPACITY)
        .expect("Deferral Count postflight trace generation must succeed");
    tester = tester.load_air_proving_ctx(Arc::new(count.0), count_ctx);
    let poseidon2_ctx = poseidon2
        .1
        .generate_proving_ctx_direct(MAX_INS_CAPACITY)
        .expect("Deferral Poseidon2 postflight trace generation must succeed");
    tester
        .load_air_proving_ctx(Arc::new(poseidon2.0), poseidon2_ctx)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_output_preflight_replay_rejects_invalid_trace_shapes() {
    assert!(super::cuda::checked_replay_trace_shape(u64::MAX, 1, usize::MAX).is_err());
    assert!(super::cuda::checked_replay_trace_shape(3, 1, 2).is_err());
    assert!(super::cuda::checked_replay_trace_shape(2, usize::MAX, 2).is_err());
}
