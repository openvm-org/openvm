use std::{array::from_fn, sync::Arc};

use openvm_circuit::arch::{
    deferral::{DeferralState, InputMapVal},
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
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::DIGEST_SIZE, p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    super::{DeferralCallAdapterRecord, DeferralCallChipGpu, DeferralCallCoreRecord},
    crate::{
        count::DeferralCircuitCountChipGpu,
        poseidon2::{poseidon2_buffer_capacity, DeferralPoseidon2ChipGpu},
    },
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        DenseRecordArena, EmptyAdapterCoreLayout,
    },
    openvm_cuda_common::d_buffer::DeviceBuffer,
};

use super::{
    DeferralCallAdapterAir, DeferralCallAdapterExecutor, DeferralCallAdapterFiller,
    DeferralCallAir, DeferralCallChip, DeferralCallCoreAir, DeferralCallCoreFiller,
    DeferralCallExecutor,
};
use crate::{
    count::{DeferralCircuitCountAir, DeferralCircuitCountBus, DeferralCircuitCountChip},
    poseidon2::{
        deferral_poseidon2_air, deferral_poseidon2_chip, DeferralPoseidon2Air,
        DeferralPoseidon2Bus, DeferralPoseidon2Chip,
    },
    utils::{
        byte_commit_to_f, join_memory_ops, COMMIT_NUM_BYTES, DIGEST_MEMORY_OPS, OUTPUT_TOTAL_BYTES,
        OUTPUT_TOTAL_MEMORY_OPS,
    },
    DeferralFn,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 1024;
const NUM_DEFERRALS: usize = 4;
const DEFERRAL_COUNT_BUS: BusIndex = 20;
const DEFERRAL_POSEIDON2_BUS: BusIndex = 21;

type Harness<RA> =
    TestChipHarness<F, DeferralCallExecutor, DeferralCallAir, DeferralCallChip<F>, RA>;
type BitwisePeriphery = (
    BitwiseOperationLookupAir<RV32_CELL_BITS>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
);
type CountPeriphery = (DeferralCircuitCountAir, Arc<DeferralCircuitCountChip>);
type Poseidon2Periphery = (DeferralPoseidon2Air<F>, Arc<DeferralPoseidon2Chip<F>>);

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    DeferralCallExecutor,
    DeferralCallAir,
    DeferralCallChipGpu,
    DeferralCallChip<F>,
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

fn deferral_fns(num_deferrals: usize) -> Vec<Arc<DeferralFn>> {
    (0..num_deferrals)
        .map(|idx| {
            Arc::new(DeferralFn::new(move |input_raw| {
                let seed = input_raw
                    .iter()
                    .fold(idx as u8, |acc, byte| acc.wrapping_add(*byte));
                let num_chunks = (seed as usize % 4) + 1;
                let len = num_chunks * DIGEST_SIZE;
                (0..len)
                    .map(|i| seed.wrapping_add(i as u8))
                    .collect::<Vec<_>>()
            }))
        })
        .collect()
}

fn read_deferral_digest(tester: &mut impl TestBuilder<F>, ptr: usize) -> [F; DIGEST_SIZE] {
    let chunks = from_fn(|chunk_idx| {
        tester
            .read::<DEFAULT_BLOCK_SIZE>(DEFERRAL_AS as usize, ptr + chunk_idx * DEFAULT_BLOCK_SIZE)
    });
    join_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(chunks)
}

fn init_streams(tester: &mut impl TestBuilder<F>, num_deferrals: usize) {
    tester.streams_mut().deferrals = vec![DeferralState::new(vec![]); num_deferrals];
}

fn set_and_execute_call<RA, E>(
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

    let input_commit_f: [F; DIGEST_SIZE] =
        from_fn(|_| F::from_u32(rng.random_range(0..F::ORDER_U32)));
    let input_commit: [u8; COMMIT_NUM_BYTES] = input_commit_f
        .iter()
        .flat_map(|x| x.as_canonical_u32().to_le_bytes())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let input_raw_len = rng.random_range(1..=(3 * DIGEST_SIZE));
    let input_raw = (0..input_raw_len)
        .map(|_| rng.random())
        .collect::<Vec<u8>>();
    tester.streams_mut().deferrals[deferral_idx].store_input(input_commit.to_vec(), input_raw);

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
    for (chunk_idx, chunk) in input_commit.chunks_exact(DEFAULT_BLOCK_SIZE).enumerate() {
        let chunk: [u8; DEFAULT_BLOCK_SIZE] = chunk.try_into().unwrap();
        tester.write(
            RV32_MEMORY_AS as usize,
            input_ptr + chunk_idx * DEFAULT_BLOCK_SIZE,
            chunk.map(F::from_u8),
        );
    }

    let input_acc_ptr = 2 * deferral_idx * DIGEST_SIZE;
    let output_acc_ptr = input_acc_ptr + DIGEST_SIZE;
    let old_input_acc = read_deferral_digest(tester, input_acc_ptr);
    let old_output_acc = read_deferral_digest(tester, output_acc_ptr);

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            DeferralOpcode::CALL.global_opcode(),
            [
                rd,
                rs,
                deferral_idx,
                RV32_REGISTER_AS as usize,
                RV32_MEMORY_AS as usize,
            ],
        ),
    );

    let (output_commit, output_raw) = {
        let state = &tester.streams_mut().deferrals[deferral_idx];
        let output_commit = match state.get_input(&input_commit.to_vec()) {
            InputMapVal::Output(output_commit) => output_commit.clone(),
            InputMapVal::Raw(_) => panic!("deferral CALL should cache the computed output"),
        };
        let output_raw = state.get_output(&output_commit).clone();
        (output_commit, output_raw)
    };

    let mut output_key = [0u8; OUTPUT_TOTAL_BYTES];
    for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
        let chunk: [F; DEFAULT_BLOCK_SIZE] = tester.read(
            RV32_MEMORY_AS as usize,
            output_ptr + chunk_idx * DEFAULT_BLOCK_SIZE,
        );
        for i in 0..DEFAULT_BLOCK_SIZE {
            output_key[chunk_idx * DEFAULT_BLOCK_SIZE + i] = chunk[i].as_canonical_u32() as u8;
        }
    }
    let output_commit_expected: [u8; COMMIT_NUM_BYTES] = output_commit.clone().try_into().unwrap();
    assert_eq!(
        &output_key[..COMMIT_NUM_BYTES],
        &output_commit_expected,
        "output commit mismatch"
    );
    assert_eq!(
        &output_key[COMMIT_NUM_BYTES..],
        &(output_raw.len() as u64).to_le_bytes(),
        "output length mismatch"
    );

    let poseidon2_chip = deferral_poseidon2_chip::<F>();
    let input_f_commit = byte_commit_to_f(&input_commit.map(F::from_u8));
    let output_f_commit = byte_commit_to_f(&output_commit_expected.map(F::from_u8));
    let expected_new_input_acc = poseidon2_chip.perm(&old_input_acc, &input_f_commit, true);
    let expected_new_output_acc = poseidon2_chip.perm(&old_output_acc, &output_f_commit, true);
    let new_input_acc = read_deferral_digest(tester, input_acc_ptr);
    let new_output_acc = read_deferral_digest(tester, output_acc_ptr);
    assert_eq!(
        new_input_acc, expected_new_input_acc,
        "input accumulator mismatch"
    );
    assert_eq!(
        new_output_acc, expected_new_output_acc,
        "output accumulator mismatch"
    );
}

fn create_cpu_harness(
    tester: &VmChipTestBuilder<F>,
    num_deferrals: usize,
    fns: Vec<Arc<DeferralFn>>,
) -> CpuHarnessBundle {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

    let air = DeferralCallAir::new(
        DeferralCallAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        DeferralCallCoreAir::new(count_bus, poseidon2_bus, bitwise_bus),
    );
    let executor = DeferralCallExecutor::new(DeferralCallAdapterExecutor, fns);
    let chip = DeferralCallChip::new(
        DeferralCallCoreFiller::new(
            DeferralCallAdapterFiller::new(bitwise_chip.clone(), tester.address_bits()),
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
fn create_cuda_harness(
    tester: &GpuChipTestBuilder,
    num_deferrals: usize,
    fns: Vec<Arc<DeferralFn>>,
) -> CudaHarnessBundle {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let count_bus = DeferralCircuitCountBus::new(DEFERRAL_COUNT_BUS);
    let poseidon2_bus = DeferralPoseidon2Bus::new(DEFERRAL_POSEIDON2_BUS);
    let count_chip_cpu = Arc::new(DeferralCircuitCountChip::new(num_deferrals));
    let poseidon2_chip_cpu = Arc::new(deferral_poseidon2_chip());

    let air = DeferralCallAir::new(
        DeferralCallAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        DeferralCallCoreAir::new(count_bus, poseidon2_bus, bitwise_bus),
    );
    let executor = DeferralCallExecutor::new(DeferralCallAdapterExecutor, fns);
    let cpu_chip = DeferralCallChip::new(
        DeferralCallCoreFiller::new(
            DeferralCallAdapterFiller::new(dummy_bitwise_chip.clone(), tester.address_bits()),
            count_chip_cpu,
            poseidon2_chip_cpu,
            dummy_bitwise_chip,
            tester.address_bits(),
        ),
        tester.dummy_memory_helper(),
    );

    let count = Arc::new(DeviceBuffer::<u32>::with_capacity(num_deferrals));
    count.fill_zero().unwrap();
    let poseidon2_chip_gpu =
        DeferralPoseidon2ChipGpu::new(poseidon2_buffer_capacity(MAX_INS_CAPACITY.max(1)), 1);
    let gpu_chip = DeferralCallChipGpu::new(
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
            DeferralCircuitCountChipGpu::new(count, num_deferrals),
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
fn rand_deferral_call_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::<F>::from_config(test_memory_config());
    let CpuHarnessBundle {
        mut harness,
        bitwise,
        count,
        poseidon2,
    } = create_cpu_harness(&tester, NUM_DEFERRALS, deferral_fns(NUM_DEFERRALS));

    init_streams(&mut tester, NUM_DEFERRALS);
    let num_ops = 25;
    for _ in 0..num_ops {
        set_and_execute_call(
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
fn test_cuda_rand_deferral_call_tracegen() {
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
    } = create_cuda_harness(&tester, NUM_DEFERRALS, deferral_fns(NUM_DEFERRALS));

    init_streams(&mut tester, NUM_DEFERRALS);
    let num_ops = 40;
    for _ in 0..num_ops {
        set_and_execute_call(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            NUM_DEFERRALS,
        );
    }

    type Record<'a> = (
        &'a mut DeferralCallAdapterRecord<F>,
        &'a mut DeferralCallCoreRecord<F>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, DeferralCallAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .load(count.0, count.1, count.2)
        .load(poseidon2.0, poseidon2.1, poseidon2.2)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
