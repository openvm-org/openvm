use std::sync::Arc;

use openvm_circuit::arch::{
    deferral::{DeferralResult, DeferralState},
    testing::{
        memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    Arena, MatrixRecordArena, MemoryConfig, PreflightExecutor, DEFAULT_BLOCK_SIZE,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::{interaction::BusIndex, p3_field::PrimeCharacteristicRing};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::DIGEST_SIZE, p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng, RngCore};
#[cfg(feature = "cuda")]
use {
    super::{DeferralOutputChipGpu, DeferralOutputRecordMut},
    crate::{count::DeferralCircuitCountChipGpu, poseidon2::DeferralPoseidon2ChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        DenseRecordArena,
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
    utils::{combine_output, COMMIT_NUM_BYTES, OUTPUT_TOTAL_BYTES},
    RawDeferralResult,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 1024;
const NUM_DEFERRALS: usize = 4;
const DEFERRAL_COUNT_BUS: BusIndex = 20;
const DEFERRAL_POSEIDON2_BUS: BusIndex = 21;

type Harness<RA> =
    TestChipHarness<F, DeferralOutputExecutor, DeferralOutputAir, DeferralOutputChip<F>, RA>;
type BitwisePeriphery = (
    BitwiseOperationLookupAir<RV32_CELL_BITS>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
);
type CountPeriphery = (DeferralCircuitCountAir, Arc<DeferralCircuitCountChip>);
type Poseidon2Periphery = (DeferralPoseidon2Air<F>, Arc<DeferralPoseidon2Chip<F>>);

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    DeferralOutputExecutor,
    DeferralOutputAir,
    DeferralOutputChipGpu,
    DeferralOutputChip<F>,
>;
#[cfg(feature = "cuda")]
type CudaCountPeriphery = (
    DeferralCircuitCountAir,
    DeferralCircuitCountChipGpu,
    DenseRecordArena,
);
#[cfg(feature = "cuda")]
type CudaPoseidon2Periphery = (
    DeferralPoseidon2Air<F>,
    DeferralPoseidon2ChipGpu,
    DenseRecordArena,
);

struct CpuHarnessBundle {
    harness: Harness<MatrixRecordArena<F>>,
    bitwise: BitwisePeriphery,
    count: CountPeriphery,
    poseidon2: Poseidon2Periphery,
}

#[cfg(feature = "cuda")]
struct CudaHarnessBundle {
    harness: GpuHarness,
    count: CudaCountPeriphery,
    poseidon2: CudaPoseidon2Periphery,
}

fn test_memory_config() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
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
    for (chunk_idx, chunk) in output_key.chunks_exact(DEFAULT_BLOCK_SIZE).enumerate() {
        let chunk: [u8; DEFAULT_BLOCK_SIZE] = chunk.try_into().unwrap();
        tester.write(
            RV32_MEMORY_AS as usize,
            input_ptr + chunk_idx * DEFAULT_BLOCK_SIZE,
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

fn set_and_execute_output<RA, E>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    num_deferrals: usize,
) where
    RA: Arena,
    E: PreflightExecutor<F, RA>,
{
    let rd = gen_pointer(rng, DEFAULT_BLOCK_SIZE);
    let rs = gen_pointer(rng, DEFAULT_BLOCK_SIZE);
    let output_ptr = gen_pointer(rng, DEFAULT_BLOCK_SIZE);
    let input_ptr = gen_pointer(rng, DEFAULT_BLOCK_SIZE);
    let deferral_idx = rng.random_range(0..num_deferrals);

    let mut input_commit = [0u8; COMMIT_NUM_BYTES];
    rng.fill_bytes(&mut input_commit);
    let output_len = rng.random_range(0..=4) * DIGEST_SIZE;
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

    tester.write(
        RV32_REGISTER_AS as usize,
        rd,
        (output_ptr as u32).to_le_bytes().map(F::from_u8),
    );
    tester.write(
        RV32_REGISTER_AS as usize,
        rs,
        (input_ptr as u32).to_le_bytes().map(F::from_u8),
    );

    let output_commit: [u8; COMMIT_NUM_BYTES] = result.output_commit.try_into().unwrap();
    let output_key = combine_output(
        output_commit,
        (result.output_raw.len() as u64).to_le_bytes(),
    );
    write_output_key(tester, input_ptr, output_key);

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            DeferralOpcode::OUTPUT.global_opcode(),
            [
                rd,
                rs,
                deferral_idx,
                RV32_REGISTER_AS as usize,
                RV32_MEMORY_AS as usize,
            ],
        ),
    );
}

fn create_cpu_harness(tester: &VmChipTestBuilder<F>, num_deferrals: usize) -> CpuHarnessBundle {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

    let air = DeferralOutputAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        count_bus,
        poseidon2_bus,
        bitwise_bus,
        tester.address_bits(),
    );
    let executor = DeferralOutputExecutor::new();
    let chip = DeferralOutputChip::new(
        DeferralOutputFiller::new(
            count_chip.clone(),
            poseidon2_chip.clone(),
            bitwise_chip.clone(),
            tester.address_bits(),
        ),
        tester.memory_helper(),
    );

    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    CpuHarnessBundle {
        harness,
        bitwise: (bitwise_chip.air, bitwise_chip),
        count: (
            DeferralCircuitCountAir::new(count_bus, num_deferrals),
            count_chip,
        ),
        poseidon2: (deferral_poseidon2_air(poseidon2_bus.0), poseidon2_chip),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
fn create_cuda_harness(tester: &GpuChipTestBuilder, num_deferrals: usize) -> CudaHarnessBundle {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip_cpu = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip_cpu = Arc::new(deferral_poseidon2_chip());

    let air = DeferralOutputAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        count_bus,
        poseidon2_bus,
        bitwise_bus,
        tester.address_bits(),
    );
    let executor = DeferralOutputExecutor::new();
    let cpu_chip = DeferralOutputChip::new(
        DeferralOutputFiller::new(
            count_chip_cpu,
            poseidon2_chip_cpu,
            dummy_bitwise_chip,
            tester.address_bits(),
        ),
        tester.dummy_memory_helper(),
    );

    let ctx = tester.range_checker().ctx.clone();
    let count = Arc::new(DeviceBuffer::<u32>::with_capacity_on(num_deferrals, &ctx));
    count.fill_zero_on(&ctx).unwrap();
    let poseidon2_chip_gpu = DeferralPoseidon2ChipGpu::new(MAX_INS_CAPACITY.max(1), 1, ctx.clone());
    let gpu_chip = DeferralOutputChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
        count.clone(),
        num_deferrals,
        poseidon2_chip_gpu.shared_buffer(),
    );

    let harness = GpuHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);
    CudaHarnessBundle {
        harness,
        count: (
            DeferralCircuitCountAir::new(count_bus, num_deferrals),
            DeferralCircuitCountChipGpu::new(count, num_deferrals, ctx),
            DenseRecordArena::with_byte_capacity(0),
        ),
        poseidon2: (
            deferral_poseidon2_air(poseidon2_bus.0),
            poseidon2_chip_gpu,
            DenseRecordArena::with_byte_capacity(0),
        ),
    }
}

#[test]
fn rand_deferral_output_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::<F>::from_config(test_memory_config());
    let CpuHarnessBundle {
        mut harness,
        bitwise,
        count,
        poseidon2,
    } = create_cpu_harness(&tester, NUM_DEFERRALS);

    init_streams(&mut tester, NUM_DEFERRALS);
    let num_ops = 25;
    for _ in 0..num_ops {
        set_and_execute_output(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            NUM_DEFERRALS,
        );
    }

    tester
        .build()
        .load(harness)
        .load_periphery(count)
        .load_periphery(poseidon2)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_deferral_output_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::new(
        test_memory_config(),
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    )
    .with_bitwise_op_lookup(default_bitwise_lookup_bus());
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
            &mut harness.dense_arena,
            &mut rng,
            NUM_DEFERRALS,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<DeferralOutputRecordMut<'_>, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .load(count.0, count.1, count.2)
        .load(poseidon2.0, poseidon2.1, poseidon2.2)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
