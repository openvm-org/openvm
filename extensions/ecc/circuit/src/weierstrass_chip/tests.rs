use std::str::FromStr;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use std::sync::Arc;

use halo2curves_axiom::secp256r1;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num, Zero};
use openvm_circuit::arch::{
    testing::{
        memory::{gen_distinct_register_pointers, gen_pointer},
        TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder,
    },
    MemoryConfig, Postflight, MEMORY_BLOCK_BYTES,
};
use openvm_circuit_primitives::bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::generate_random_biguint, utils::biguint_to_limbs_vec, ExprBuilderConfig,
};
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::extension::HybridWeierstrassChip,
    openvm_circuit::arch::testing::{
        default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    openvm_circuit::arch::cuda::postflight::GpuPostflightProgram,
    openvm_circuit::arch::rvr::PreflightLimits,
    openvm_circuit::system::cuda::memory::MemoryInventoryGPU,
    openvm_circuit::{
        arch::{
            PreflightHistory, PreflightMemoryLog, PreflightProgramEvent, VirtualMachine, VmExecutor,
        },
        utils::{test_gpu_engine, test_system_config},
    },
    openvm_cuda_common::copy::MemCopyD2H,
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        program::Program,
        SystemOpcode,
    },
    openvm_mod_circuit_builder::{run_field_expression_precomputed, FieldExpressionProgram},
    openvm_stark_backend::StarkEngine,
    rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PREFLIGHT_WRITE_BIT},
    strum::EnumCount,
};

use crate::{
    get_ec_add_air, get_ec_add_chip, get_ec_add_executor, get_ec_double_air, get_ec_double_chip,
    get_ec_double_executor,
    weierstrass_chip::{
        generate_add_trace_from_postflight, generate_add_trace_from_postflights,
        generate_double_trace_from_postflight, generate_double_trace_from_postflights,
    },
    EcDoubleExecutor, WeierstrassAir, WeierstrassChip, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_48,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

lazy_static::lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        let x1 = BigUint::from_u32(1).unwrap();
        let y1 = BigUint::from_str(
            "29896722852569046015560700294576055776214335159245303116488692907525646231534",
        )
        .unwrap();
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "69211104694897500952317515077652022726490027694212560352756646854116994689233",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str(
            "109562500687829935604265064386702914290271628241900466384583316550888437213118",
        )
        .unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331",
        )
        .unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str(
            "88733411122275068320336854419305339160905807011607464784153110222112026831518",
        )
        .unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn packed_u16_block(bytes: &[u8]) -> [u16; 4] {
    assert_eq!(bytes.len(), MEMORY_BLOCK_BYTES);
    std::array::from_fn(|index| u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]))
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn encode_field_inputs(values: &[BigUint], num_limbs: usize) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| biguint_to_limbs_vec(value, num_limbs))
        .collect()
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn make_vec_heap_history<const NUM_READS: usize, const BLOCKS: usize>(
    instruction: Instruction<F>,
    rs_ptrs: [u32; NUM_READS],
    rd_ptr: u32,
    rs_vals: [u32; NUM_READS],
    rd_val: u32,
    input_bytes: &[u8],
    output_bytes: &[u8],
) -> (Program<F>, PreflightHistory) {
    let bytes_per_value = BLOCKS * MEMORY_BLOCK_BYTES;
    assert_eq!(input_bytes.len(), NUM_READS * bytes_per_value);
    assert_eq!(output_bytes.len(), bytes_per_value);
    let event_count = NUM_READS + 1 + NUM_READS * BLOCKS + BLOCKS;
    let final_timestamp = 1 + event_count as u32;
    let register_block = |pointer: u32| [pointer as u16, (pointer >> 16) as u16, 0, 0];
    let mut timestamp = 1u32;
    let mut memory_log = Vec::with_capacity(event_count);
    for (&register, &pointer) in rs_ptrs.iter().zip(&rs_vals) {
        memory_log.push(PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: REGISTER_AS,
            pointer: register / 2,
            value: register_block(pointer),
        });
        timestamp += 1;
    }
    memory_log.push(PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: REGISTER_AS,
        pointer: rd_ptr / 2,
        value: register_block(rd_val),
    });
    timestamp += 1;
    for (read, &pointer) in rs_vals.iter().enumerate() {
        for block in 0..BLOCKS {
            let start = read * bytes_per_value + block * MEMORY_BLOCK_BYTES;
            memory_log.push(PreflightMemoryEvent {
                timestamp,
                address_space_and_kind: MEMORY_AS,
                pointer: pointer / 2 + (block * 4) as u32,
                value: packed_u16_block(&input_bytes[start..start + MEMORY_BLOCK_BYTES]),
            });
            timestamp += 1;
        }
    }
    let mut initial_write_log = Vec::with_capacity(BLOCKS);
    for block in 0..BLOCKS {
        let start = block * MEMORY_BLOCK_BYTES;
        let pointer = rd_val / 2 + (block * 4) as u32;
        memory_log.push(PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: MEMORY_AS | PREFLIGHT_WRITE_BIT,
            pointer,
            value: packed_u16_block(&output_bytes[start..start + MEMORY_BLOCK_BYTES]),
        });
        initial_write_log.push(PreflightInitialWrite {
            address_space: MEMORY_AS,
            pointer,
            initial_value: [0; 4],
        });
        timestamp += 1;
    }
    assert_eq!(timestamp, final_timestamp);

    let terminate =
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
    (
        Program::from_instructions(&[instruction, terminate]),
        PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: final_timestamp,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: final_timestamp,
                },
            ],
            memory: PreflightMemoryLog {
                accesses: memory_log,
                initial_writes: initial_write_log,
                ..Default::default()
            },
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn repeat_vec_heap_history(
    instruction: Instruction<F>,
    history: PreflightHistory,
    repetitions: usize,
) -> (Program<F>, PreflightHistory) {
    assert!(repetitions > 0);
    let first_timestamp = history.program[0].timestamp;
    let timestamp_step = history.program[1].timestamp - first_timestamp;
    let mut memory_log = Vec::with_capacity(history.memory.accesses.len() * repetitions);
    let mut program_log = Vec::with_capacity(repetitions + 2);
    for repetition in 0..repetitions {
        let timestamp_shift = repetition as u32 * timestamp_step;
        program_log.push(PreflightProgramEvent {
            pc: repetition as u32 * 4,
            timestamp: first_timestamp + timestamp_shift,
        });
        memory_log.extend(history.memory.accesses.iter().copied().map(|mut event| {
            event.timestamp += timestamp_shift;
            event
        }));
    }
    let final_pc = repetitions as u32 * 4;
    let final_timestamp = first_timestamp + repetitions as u32 * timestamp_step;
    program_log.extend([
        PreflightProgramEvent {
            pc: final_pc,
            timestamp: final_timestamp,
        },
        PreflightProgramEvent {
            pc: final_pc,
            timestamp: final_timestamp,
        },
    ]);
    let terminate =
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
    let mut instructions = vec![instruction; repetitions];
    instructions.push(terminate);
    (
        Program::from_instructions(&instructions),
        PreflightHistory {
            program: program_log,
            memory: PreflightMemoryLog {
                accesses: memory_log,
                initial_writes: history.memory.initial_writes,
                ..Default::default()
            },
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn field_expression_output(
    program: &FieldExpressionProgram,
    input_bytes: &[u8],
    is_setup: bool,
) -> Vec<u8> {
    let flag = if is_setup { program.num_flags() } else { 0 };
    run_field_expression_precomputed::<true>(program, flag, input_bytes).0
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn reset_gpu_initial_memory(tester: &mut GpuChipTestBuilder) {
    tester.memory.memory.data.memory.recompute_touched_pages();
    let device_ctx = tester.range_checker().device_ctx.clone();
    let hasher_chip = tester.memory.hasher_chip.clone().unwrap();
    tester.memory.inventory =
        MemoryInventoryGPU::new(tester.memory.config.clone(), hasher_chip, device_ctx);
    tester
        .memory
        .inventory
        .set_initial_memory(&tester.memory.memory.data.memory);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn initialize_vec_heap_memory<const NUM_READS: usize, const BLOCKS: usize>(
    tester: &mut GpuChipTestBuilder,
    rs_ptrs: [u32; NUM_READS],
    rd_ptr: u32,
    rs_vals: [u32; NUM_READS],
    rd_val: u32,
    input_bytes: &[u8],
) {
    let bytes_per_value = BLOCKS * MEMORY_BLOCK_BYTES;
    for (&register, &pointer) in rs_ptrs.iter().zip(&rs_vals) {
        unsafe {
            tester.memory.memory.data.write::<u16, 4>(
                REGISTER_AS,
                register / 2,
                [pointer as u16, (pointer >> 16) as u16, 0, 0],
            );
        }
    }
    unsafe {
        tester.memory.memory.data.write::<u16, 4>(
            REGISTER_AS,
            rd_ptr / 2,
            [rd_val as u16, (rd_val >> 16) as u16, 0, 0],
        );
    }
    for (read, &pointer) in rs_vals.iter().enumerate() {
        for block in 0..BLOCKS {
            let start = read * bytes_per_value + block * MEMORY_BLOCK_BYTES;
            unsafe {
                tester.memory.memory.data.write::<u16, 4>(
                    MEMORY_AS,
                    pointer / 2 + (block * 4) as u32,
                    packed_u16_block(&input_bytes[start..start + MEMORY_BLOCK_BYTES]),
                );
            }
        }
    }
    for block in 0..BLOCKS {
        unsafe {
            tester.memory.memory.data.write::<u16, 4>(
                MEMORY_AS,
                rd_val / 2 + (block * 4) as u32,
                [0; 4],
            );
        }
    }
    reset_gpu_initial_memory(tester);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn gpu_range_counts(tester: &GpuChipTestBuilder) -> Vec<u32> {
    tester
        .range_checker()
        .count
        .to_host_on(&tester.range_checker().device_ctx)
        .unwrap()
        .into_iter()
        // GPU lookup histograms store ordinary u32 counters in an F-sized buffer.
        // SAFETY: BabyBear and u32 have the same representation size, as required by the GPU
        // variable range checker.
        .map(|count| unsafe { std::mem::transmute::<F, u32>(count) })
        .collect()
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn combine_two_vec_heap_histories(
    first_instruction: Instruction<F>,
    mut first: PreflightHistory,
    second_instruction: Instruction<F>,
    mut second: PreflightHistory,
) -> (Program<F>, PreflightHistory) {
    let second_start = first.program[1].timestamp;
    let timestamp_shift = second_start - second.program[0].timestamp;
    for event in &mut second.memory.accesses {
        event.timestamp += timestamp_shift;
    }
    let second_end = second.program[1].timestamp + timestamp_shift;
    first.memory.accesses.extend(second.memory.accesses);
    first
        .memory
        .initial_writes
        .extend(second.memory.initial_writes);
    first.program = vec![
        first.program[0],
        PreflightProgramEvent {
            pc: 4,
            timestamp: second_start,
        },
        PreflightProgramEvent {
            pc: 8,
            timestamp: second_end,
        },
        PreflightProgramEvent {
            pc: 8,
            timestamp: second_end,
        },
    ];
    let terminate =
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
    (
        Program::from_instructions(&[first_instruction, second_instruction, terminate]),
        first,
    )
}

mod ec_add_tests {
    use num_traits::One;

    use super::*;
    use crate::EcAddExecutor;

    type EcAddHarness<const BLOCKS: usize> = TestChipHarness<
        F,
        EcAddExecutor<BLOCKS>,
        WeierstrassAir<2, BLOCKS>,
        WeierstrassChip<F, 2, BLOCKS>,
    >;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
        b: BigUint,
    ) -> EcAddHarness<BLOCKS> {
        let air = get_ec_add_air::<BLOCKS>(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );
        let executor = get_ec_add_executor::<BLOCKS>(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            offset,
            a.clone(),
            b.clone(),
        );
        let chip = get_ec_add_chip::<F, BLOCKS>(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
            a,
            b,
        );

        EcAddHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| generate_add_trace_from_postflight(chip, postflight, offset),
        )
        .with_batch_trace_generator(move |chip, postflights| {
            generate_add_trace_from_postflights(chip, postflights, offset)
        })
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        EcAddExecutor<BLOCKS>,
        WeierstrassAir<2, BLOCKS>,
        HybridWeierstrassChip<F, 2, BLOCKS>,
        WeierstrassChip<F, 2, BLOCKS>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
        b: BigUint,
    ) -> GpuHarness<BLOCKS> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_ec_add_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
            a.clone(),
            b.clone(),
        );
        let executor = get_ec_add_executor(
            config.clone(),
            range_bus.range_max_bits,
            offset,
            a.clone(),
            b.clone(),
        );

        let cpu_chip = get_ec_add_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
            a.clone(),
            b.clone(),
        );

        let gpu_cpu_chip = get_ec_add_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
            a,
            b,
        );
        #[cfg(feature = "rvr")]
        let hybrid_chip = HybridWeierstrassChip::new_with_replay(
            gpu_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            offset,
            tester.range_checker(),
        )
        .unwrap();
        #[cfg(not(feature = "rvr"))]
        let hybrid_chip =
            HybridWeierstrassChip::new(gpu_cpu_chip, tester.range_checker().device_ctx.clone());

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_add_trace_from_postflight(chip, postflight, offset)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
            .with_batch_trace_generator(move |chip, postflights| {
                generate_add_trace_from_postflights(chip, postflights, offset)
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_add<const BLOCKS: usize, const NUM_LIMBS: usize>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcAddExecutor<BLOCKS>,
        preflight: &mut TestPreflight<F>,
        rng: &mut StdRng,
        modulus: &BigUint,
        a: &BigUint,
        b: &BigUint,
        is_setup: bool,
        offset: usize,
        p1: Option<(BigUint, BigUint)>,
        p2: Option<(BigUint, BigUint)>,
    ) -> Instruction<F> {
        // For projective coordinates, each point has 3 coordinates (X, Y, Z).
        // For setup: P1 = (modulus, a, b), P2 = (1, 1, 1) (dummy).
        // For normal: P1 = (x1, y1, 1), P2 = (x2, y2, 1) (affine embedded as Z = 1).
        let (x1, y1, z1, x2, y2, z2, op_local) = if is_setup {
            (
                modulus.clone(),
                a.clone(),
                b.clone(),
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
                WeierstrassOpcode::SETUP_SW_EC_ADD_PROJ as usize,
            )
        } else if let Some((px1, py1)) = p1 {
            let (px2, py2) = p2.unwrap();
            let px1 = px1 % modulus;
            let py1 = py1 % modulus;
            let px2 = px2 % modulus;
            let py2 = py2 % modulus;
            let one = BigUint::one();
            // Complete addition formulas handle all input pairs, so no x1 != x2 special-casing.
            if rng.random_bool(0.5) {
                (
                    px1,
                    py1,
                    one.clone(),
                    px2,
                    py2,
                    one,
                    WeierstrassOpcode::SW_EC_ADD_PROJ as usize,
                )
            } else {
                (
                    px2,
                    py2,
                    one.clone(),
                    px1,
                    py1,
                    one,
                    WeierstrassOpcode::SW_EC_ADD_PROJ as usize,
                )
            }
        } else {
            panic!("Generating random inputs generically is harder because the input points need to be on the curve.");
        };

        let ptr_as = REGISTER_AS as usize;
        let data_as = MEMORY_AS as usize;

        let [rs1_ptr, rs2_ptr, rd_ptr] = gen_distinct_register_pointers(rng, REGISTER_NUM_LIMBS);

        let p1_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let p2_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let result_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;

        tester.write_bytes::<REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<REGISTER_NUM_LIMBS>(
            ptr_as,
            rs2_ptr,
            p2_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<REGISTER_NUM_LIMBS>(
            ptr_as,
            rd_ptr,
            result_base_addr.to_le_bytes().map(F::from_u8),
        );

        let x1_limbs: Vec<F> = biguint_to_limbs_vec(&x1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let z1_limbs: Vec<F> = biguint_to_limbs_vec(&z1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let x2_limbs: Vec<F> = biguint_to_limbs_vec(&x2, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let y2_limbs: Vec<F> = biguint_to_limbs_vec(&y2, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let z2_limbs: Vec<F> = biguint_to_limbs_vec(&z2, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        // Write projective points P1 = (X1, Y1, Z1) and P2 = (X2, Y2, Z2), each coordinate
        // occupying NUM_LIMBS bytes and written a memory block at a time.
        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                p1_base_addr as usize + i,
                x1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p1_base_addr + NUM_LIMBS as u64) as usize + i,
                y1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p1_base_addr + 2 * NUM_LIMBS as u64) as usize + i,
                z1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );

            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                p2_base_addr as usize + i,
                x2_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p2_base_addr + NUM_LIMBS as u64) as usize + i,
                y2_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p2_base_addr + 2 * NUM_LIMBS as u64) as usize + i,
                z2_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            rs2_ptr as isize,
            ptr_as as isize,
            data_as as isize,
        );

        tester.execute(executor, preflight, &instruction);
        instruction
    }

    fn run_ec_add_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            true,
            offset,
            None,
            None,
        );

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].clone()),
            Some(SampleEcPoints[1].clone()),
        );

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
        );

        let tester = tester.build().load(harness).finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_ec_add_32limb() {
        // secp256k1: a=0, b=7
        run_ec_add_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[test]
    fn test_ec_add_48limb() {
        // BLS12-381: a=0, b=4
        run_ec_add_test::<{ ECC_BLOCKS_48 }, { NUM_LIMBS_48 }>(
            WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[cfg(feature = "cuda")]
    fn run_cuda_ec_add<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness =
            create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            true,
            offset,
            None,
            None,
        );

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].clone()),
            Some(SampleEcPoints[1].clone()),
        );

        set_and_execute_ec_add::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
        );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_weierstrass_add_cuda_2x32() {
        run_cuda_ec_add::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_weierstrass_add_cuda_6x16() {
        run_cuda_ec_add::<ECC_BLOCKS_48, NUM_LIMBS_48>(
            WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_preflight_ec_add<const BLOCKS: usize, const NUM_LIMBS: usize>(
        modulus: BigUint,
        a: BigUint,
        b: BigUint,
        is_setup: bool,
    ) {
        let offset = WeierstrassOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let mut tester = GpuChipTestBuilder::default();
        let harness = create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());
        let values = if is_setup {
            vec![
                modulus,
                a,
                b,
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
            ]
        } else {
            vec![
                BigUint::from(1u8),
                BigUint::from(2u8),
                BigUint::one(),
                BigUint::from(3u8),
                BigUint::from(4u8),
                BigUint::one(),
            ]
        };
        let input_bytes = encode_field_inputs(&values, NUM_LIMBS);
        let output_bytes =
            field_expression_output(harness.executor.program(), &input_bytes, is_setup);
        let rs_ptrs = [16u32, 24];
        let rd_ptr = 8u32;
        let rs_vals = [0x100u32, 0x200];
        let rd_val = 0x300u32;
        initialize_vec_heap_memory::<2, BLOCKS>(
            &mut tester,
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
        );
        let local_opcode = if is_setup {
            WeierstrassOpcode::SETUP_SW_EC_ADD_PROJ
        } else {
            WeierstrassOpcode::SW_EC_ADD_PROJ
        };
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + local_opcode as usize),
            [
                rd_ptr as usize,
                rs_ptrs[0] as usize,
                rs_ptrs[1] as usize,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let device_ctx = tester.range_checker().device_ctx.clone();
        let (program, mut history) = make_vec_heap_history::<2, BLOCKS>(
            instruction,
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
            &output_bytes,
        );
        let valid_history = history.clone();
        tester.record_preflight_history(&program, &valid_history, Some(0));
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        let replay_ctx = harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
            .unwrap();
        let replay_counts = gpu_range_counts(&tester);

        let write_start = 3 + 2 * BLOCKS;
        history.memory.accesses[write_start].value[0] ^= 1;
        let (corrupt_transcript, corrupt_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        assert!(harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &corrupt_transcript, &corrupt_plan)
            .is_err());
        assert_eq!(replay_counts, gpu_range_counts(&tester));

        let mut tester = tester.build();
        tester.balance_preflight_history(&program, &valid_history, Some(0));
        tester
            .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
            .finalize()
            .simple_test()
            .expect("Weierstrass add postflight proof failed");
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_add_preflight_rows_counts_setup_and_corruption_32_48() {
        for is_setup in [false, true] {
            run_preflight_ec_add::<ECC_BLOCKS_32, NUM_LIMBS_32>(
                secp256k1_coord_prime(),
                BigUint::zero(),
                BigUint::from(7u8),
                is_setup,
            );
            run_preflight_ec_add::<ECC_BLOCKS_48, NUM_LIMBS_48>(
                BLS12_381_MODULUS.clone(),
                BigUint::zero(),
                BigUint::from(4u8),
                is_setup,
            );
        }
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_coordinator_distinguishes_repeated_curve_instances() {
        let first_base = WeierstrassOpcode::CLASS_OFFSET;
        let second_base = first_base + WeierstrassOpcode::COUNT;
        let config = ExprBuilderConfig {
            modulus: secp256k1_coord_prime(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let tester = GpuChipTestBuilder::default();
        let a = BigUint::zero();
        let b = BigUint::from(7u8);
        let first = create_cuda_harness::<ECC_BLOCKS_32>(
            &tester,
            config.clone(),
            first_base,
            a.clone(),
            b.clone(),
        );
        let second = create_cuda_harness::<ECC_BLOCKS_32>(&tester, config, second_base, a, b);

        let first_input = encode_field_inputs(
            &[
                BigUint::from(1u8),
                BigUint::from(2u8),
                BigUint::one(),
                BigUint::from(3u8),
                BigUint::from(4u8),
                BigUint::one(),
            ],
            NUM_LIMBS_32,
        );
        let second_input = encode_field_inputs(
            &[
                BigUint::from(5u8),
                BigUint::from(6u8),
                BigUint::one(),
                BigUint::from(7u8),
                BigUint::from(8u8),
                BigUint::one(),
            ],
            NUM_LIMBS_32,
        );
        let first_output = field_expression_output(first.executor.program(), &first_input, false);
        let second_output =
            field_expression_output(second.executor.program(), &second_input, false);
        let first_instruction = Instruction::from_usize(
            VmOpcode::from_usize(first_base + WeierstrassOpcode::SW_EC_ADD_PROJ as usize),
            [8, 16, 24, REGISTER_AS as usize, MEMORY_AS as usize],
        );
        let second_instruction = Instruction::from_usize(
            VmOpcode::from_usize(second_base + WeierstrassOpcode::SW_EC_ADD_PROJ as usize),
            [32, 40, 48, REGISTER_AS as usize, MEMORY_AS as usize],
        );
        let (_, first_history) = make_vec_heap_history::<2, ECC_BLOCKS_32>(
            first_instruction.clone(),
            [16, 24],
            8,
            [0x100, 0x200],
            0x300,
            &first_input,
            &first_output,
        );
        let (_, second_history) = make_vec_heap_history::<2, ECC_BLOCKS_32>(
            second_instruction.clone(),
            [40, 48],
            32,
            [0x400, 0x500],
            0x600,
            &second_input,
            &second_output,
        );
        let (program, history) = combine_two_vec_heap_histories(
            first_instruction,
            first_history,
            second_instruction,
            second_history,
        );
        let device_ctx = tester.range_checker().device_ctx.clone();
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        let extension = crate::WeierstrassExtension::new(vec![
            crate::SECP256K1_CONFIG.clone(),
            crate::SECP256K1_CONFIG.clone(),
        ]);
        let mut incomplete = crate::WeierstrassPreflightGpuTracegen::new(
            &extension,
            &gpu_program,
            &gpu_transcript,
            &replay_plan,
        );
        assert!(incomplete.generate_for_chip(&()).unwrap().is_none());
        drop(
            incomplete
                .generate_for_chip(&first.gpu_chip)
                .unwrap()
                .unwrap(),
        );
        let error = incomplete
            .finish()
            .expect_err("the second curve's opcode must remain unclaimed");
        assert!(error
            .to_string()
            .contains(&(second_base as u32).to_string()));

        let mut complete = crate::WeierstrassPreflightGpuTracegen::new(
            &extension,
            &gpu_program,
            &gpu_transcript,
            &replay_plan,
        );
        drop(
            complete
                .generate_for_chip(&first.gpu_chip)
                .unwrap()
                .unwrap(),
        );
        drop(
            complete
                .generate_for_chip(&second.gpu_chip)
                .unwrap()
                .unwrap(),
        );
        complete.finish().unwrap();

        // Exercise the actual reverse inventory walk and its ECC -> Algebra -> RV64 fallthrough
        // on the same history. Both curve instances have the same concrete chip types;
        // only their opcode bases distinguish them.
        let mut init_memory = SparseMemoryImage::default();
        for (register, pointer) in [
            (8u32, 0x300u32),
            (16, 0x100),
            (24, 0x200),
            (32, 0x600),
            (40, 0x400),
            (48, 0x500),
        ] {
            init_memory.extend(
                u64::from(pointer)
                    .to_le_bytes()
                    .into_iter()
                    .enumerate()
                    .map(|(offset, byte)| ((REGISTER_AS, register + offset as u32), byte)),
            );
        }
        let bytes_per_value = ECC_BLOCKS_32 * MEMORY_BLOCK_BYTES;
        for (pointer, bytes) in [
            (0x100u32, &first_input[..bytes_per_value]),
            (0x200u32, &first_input[bytes_per_value..]),
            (0x400u32, &second_input[..bytes_per_value]),
            (0x500u32, &second_input[bytes_per_value..]),
        ] {
            init_memory.extend(
                bytes
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(offset, byte)| ((MEMORY_AS, pointer + offset as u32), byte)),
            );
        }
        let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
        let mut vm_config = crate::Rv64WeierstrassConfig::new(vec![
            crate::SECP256K1_CONFIG.clone(),
            crate::SECP256K1_CONFIG.clone(),
        ]);
        *vm_config.as_mut() = test_system_config();
        let executor = VmExecutor::new(vm_config.clone()).unwrap();
        let checkpoint = executor.preflight_instance(&exe).unwrap();
        let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let execution = checkpoint
            .execute(
                Vec::<Vec<u8>>::new(),
                PreflightLimits::new(3, 2 * ECC_BLOCKS_32, 1),
            )
            .unwrap();

        let mut incomplete_config =
            crate::Rv64WeierstrassConfig::new(vec![crate::SECP256K1_CONFIG.clone()]);
        *incomplete_config.as_mut() = test_system_config();
        let (mut poisoned_vm, _) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            crate::Rv64WeierstrassHybridBuilder,
            incomplete_config.clone(),
        )
        .unwrap();
        let cached_program = poisoned_vm.commit_program_on_device(&program);
        poisoned_vm.load_program(cached_program);
        poisoned_vm.transport_init_memory_to_device(&state.memory);
        let poisoned_gpu_program =
            crate::WeierstrassPreflightGpuTracegen::upload_postflight_program(
                &program,
                &incomplete_config.modular.system.memory_config,
                &incomplete_config.modular.modular,
                None,
                &extension,
                &poisoned_vm.engine.device().device_ctx,
            )
            .unwrap();
        let (poisoned_transcript, poisoned_plan) =
            crate::WeierstrassPreflightGpuTracegen::postflight(
                &poisoned_vm,
                &poisoned_gpu_program,
                &execution,
                execution.retired,
            )
            .unwrap();
        let late_coverage_error = crate::WeierstrassPreflightGpuTracegen::new(
            &extension,
            poisoned_gpu_program.program(),
            &poisoned_transcript,
            &poisoned_plan,
        )
        .generate_proving_ctx(&mut poisoned_vm, &incomplete_config.modular.modular, None)
        .err()
        .expect("the VM inventory omits the second configured curve");
        assert!(late_coverage_error
            .to_string()
            .contains(&(second_base as u32).to_string()));
        let retry_error = crate::WeierstrassPreflightGpuTracegen::new(
            &extension,
            poisoned_gpu_program.program(),
            &poisoned_transcript,
            &poisoned_plan,
        )
        .generate_proving_ctx(&mut poisoned_vm, &incomplete_config.modular.modular, None)
        .err()
        .expect("a failed preflight tracegen session must poison retries");
        assert!(matches!(
            retry_error,
            openvm_circuit::arch::GenerationError::ProverPoisoned
        ));

        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            crate::Rv64WeierstrassHybridBuilder,
            vm_config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let vm_gpu_program = crate::WeierstrassPreflightGpuTracegen::upload_postflight_program(
            &program,
            &vm_config.modular.system.memory_config,
            &vm_config.modular.modular,
            None,
            &vm_config.weierstrass,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (vm_transcript, vm_plan) = crate::WeierstrassPreflightGpuTracegen::postflight(
            &vm,
            &vm_gpu_program,
            &execution,
            execution.retired,
        )
        .unwrap();
        let proving_ctx = crate::WeierstrassPreflightGpuTracegen::new(
            &vm_config.weierstrass,
            vm_gpu_program.program(),
            &vm_transcript,
            &vm_plan,
        )
        .generate_proving_ctx(&mut vm, &vm_config.modular.modular, None)
        .unwrap();
        drop(vm_plan);
        drop(vm_transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();

        // Tracegen consumes the segment-start upload. This replay only checks that successful
        // coordination clears the session poison, so restore the fixture's input image first.
        vm.transport_init_memory_to_device(&state.memory);
        let (retry_transcript, retry_plan) = crate::WeierstrassPreflightGpuTracegen::postflight(
            &vm,
            &vm_gpu_program,
            &execution,
            execution.retired,
        )
        .unwrap();
        let retry_ctx = crate::WeierstrassPreflightGpuTracegen::new(
            &vm_config.weierstrass,
            vm_gpu_program.program(),
            &retry_transcript,
            &retry_plan,
        )
        .generate_proving_ctx(&mut vm, &vm_config.modular.modular, None)
        .expect("successful outer coordination must permit another preflight tracegen session");
        drop(retry_ctx);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // SANITY TESTS
    //
    // Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////

    /// Helper to convert projective (X, Y, Z) to affine (x, y) via x = X/Z, y = Y/Z.
    fn proj_to_affine(
        x_proj: &BigUint,
        y_proj: &BigUint,
        z_proj: &BigUint,
        p: &BigUint,
    ) -> (BigUint, BigUint) {
        // Compute z^{-1} mod p using Fermat's little theorem: z^{-1} = z^{p-2} mod p.
        let z_inv = z_proj.modpow(&(p - BigUint::from(2u32)), p);
        let x_affine = (x_proj * &z_inv) % p;
        let y_affine = (y_proj * &z_inv) % p;
        (x_affine, y_affine)
    }

    #[test]
    fn ec_add_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256k1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };

        // secp256k1: a=0, b=7
        let executor = get_ec_add_executor::<{ ECC_BLOCKS_32 }>(
            config,
            tester.range_checker().bus().range_max_bits,
            WeierstrassOpcode::CLASS_OFFSET,
            BigUint::zero(),
            BigUint::from(7u32),
        );

        let (p1_x, p1_y) = SampleEcPoints[0].clone();
        let (p2_x, p2_y) = SampleEcPoints[1].clone();

        // Projective input: (X1, Y1, Z1, X2, Y2, Z2) where Z=1 for affine points.
        let z = BigUint::one();
        let vars = executor
            .program()
            .execute(&[p1_x, p1_y, z.clone(), p2_x, p2_y, z.clone()], &[true]);
        // Output vars (X3, Y3, Z3) are the final three variables.
        let r = &vars[vars.len() - 3..];

        assert_eq!(r.len(), 3); // X3, Y3, Z3
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[2].0);
        assert_eq!(y3_affine, SampleEcPoints[2].1);

        let (p1_x, p1_y) = SampleEcPoints[2].clone();
        let (p2_x, p2_y) = SampleEcPoints[3].clone();
        let vars = executor
            .program()
            .execute(&[p1_x, p1_y, z.clone(), p2_x, p2_y, z], &[true]);
        // Output vars (X3, Y3, Z3) are the final three variables.
        let r = &vars[vars.len() - 3..];

        assert_eq!(r.len(), 3); // X3, Y3, Z3
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[4].0);
        assert_eq!(y3_affine, SampleEcPoints[4].1);
    }
}

mod ec_double_tests {
    use num_traits::One;

    use super::*;

    type EcDoubleHarness<const BLOCKS: usize> = TestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS>,
        WeierstrassAir<1, BLOCKS>,
        WeierstrassChip<F, 1, BLOCKS>,
    >;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
        b: BigUint,
    ) -> EcDoubleHarness<BLOCKS> {
        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b.clone(),
        );
        let executor = get_ec_double_executor(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            offset,
            a_biguint.clone(),
            b.clone(),
        );
        let chip = get_ec_double_chip(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
            a_biguint,
            b,
        );
        EcDoubleHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| generate_double_trace_from_postflight(chip, postflight, offset),
        )
        .with_batch_trace_generator(move |chip, postflights| {
            generate_double_trace_from_postflights(chip, postflights, offset)
        })
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS>,
        WeierstrassAir<1, BLOCKS>,
        HybridWeierstrassChip<F, 1, BLOCKS>,
        WeierstrassChip<F, 1, BLOCKS>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
        b: BigUint,
    ) -> GpuHarness<BLOCKS> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
            b.clone(),
        );
        let executor = get_ec_double_executor(
            config.clone(),
            range_bus.range_max_bits,
            offset,
            a_biguint.clone(),
            b.clone(),
        );

        let cpu_chip = get_ec_double_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
            a_biguint.clone(),
            b.clone(),
        );
        let gpu_cpu_chip = get_ec_double_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
            a_biguint,
            b,
        );
        #[cfg(feature = "rvr")]
        let hybrid_chip = HybridWeierstrassChip::new_with_replay(
            gpu_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            offset,
            tester.range_checker(),
        )
        .unwrap();
        #[cfg(not(feature = "rvr"))]
        let hybrid_chip =
            HybridWeierstrassChip::new(gpu_cpu_chip, tester.range_checker().device_ctx.clone());

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_double_trace_from_postflight(chip, postflight, offset)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
            .with_batch_trace_generator(move |chip, postflights| {
                generate_double_trace_from_postflights(chip, postflights, offset)
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_double<const BLOCKS: usize, const NUM_LIMBS: usize>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcDoubleExecutor<BLOCKS>,
        preflight: &mut TestPreflight<F>,
        rng: &mut StdRng,
        modulus: &BigUint,
        a_biguint: &BigUint,
        b_biguint: &BigUint,
        is_setup: bool,
        offset: usize,
        x: Option<BigUint>,
        y: Option<BigUint>,
    ) -> Instruction<F> {
        // For projective coordinates, each point has 3 coordinates (X, Y, Z).
        // For setup: P = (modulus, a, b).
        // For normal: P = (x, y, 1) (affine embedded as Z = 1).
        let (x1, y1, z1, op_local) = if is_setup {
            (
                modulus.clone(),
                a_biguint.clone(),
                b_biguint.clone(),
                WeierstrassOpcode::SETUP_SW_EC_DOUBLE_PROJ as usize,
            )
        } else if let Some(x) = x {
            let y = y.unwrap();
            let x = x % modulus;
            let y = y % modulus;
            (
                x,
                y,
                BigUint::one(),
                WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
            )
        } else {
            let x = generate_random_biguint(modulus);
            let y = generate_random_biguint(modulus);

            (
                x,
                y,
                BigUint::one(),
                WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
            )
        };

        let ptr_as = REGISTER_AS as usize;
        let data_as = MEMORY_AS as usize;

        let [rs1_ptr, rd_ptr] = gen_distinct_register_pointers(rng, REGISTER_NUM_LIMBS);

        let p1_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let result_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;

        tester.write_bytes::<REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<REGISTER_NUM_LIMBS>(
            ptr_as,
            rd_ptr,
            result_base_addr.to_le_bytes().map(F::from_u8),
        );

        let x1_limbs: Vec<F> = biguint_to_limbs_vec(&x1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let z1_limbs: Vec<F> = biguint_to_limbs_vec(&z1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        // Write projective point P = (X, Y, Z), each coordinate occupying NUM_LIMBS bytes.
        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                p1_base_addr as usize + i,
                x1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p1_base_addr + NUM_LIMBS as u64) as usize + i,
                y1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p1_base_addr + 2 * NUM_LIMBS as u64) as usize + i,
                z1_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            0,
            ptr_as as isize,
            data_as as isize,
        );

        tester.execute(executor, preflight, &instruction);
        instruction
    }

    fn run_ec_double_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());

        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                &a,
                &b,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[1].0.clone()),
            Some(SampleEcPoints[1].1.clone()),
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(p1_x),
            Some(p1_y),
        );

        let tester = tester.build().load(harness).finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_ec_double_32limb() {
        // secp256k1: a=0, b=7
        run_ec_double_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[test]
    fn test_ec_double_32limb_nonzero_a() {
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        run_ec_double_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
            b,
        );
    }

    #[test]
    fn test_ec_double_48limb() {
        // BLS12-381: a=0, b=4
        run_ec_double_test::<{ ECC_BLOCKS_48 }, { NUM_LIMBS_48 }>(
            WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[test]
    fn ec_double_postflight_generation() {
        let mut tester = VmChipTestBuilder::<F>::default();
        let modulus = secp256k1_coord_prime();
        let a = BigUint::zero();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let opcode_base = WeierstrassOpcode::CLASS_OFFSET;
        let b = BigUint::from(7u32);
        let mut harness =
            create_harness::<ECC_BLOCKS_32>(&tester, config, opcode_base, a.clone(), b);

        let rd_register = 16usize;
        let input_register = 8usize;
        let rd_pointer = 0x200u32;
        let input_pointer = 0x100u32;
        for (register, pointer) in [(rd_register, rd_pointer), (input_register, input_pointer)] {
            unsafe {
                tester.memory.memory.data.write_bytes(
                    REGISTER_AS,
                    register as u32,
                    u64::from(pointer).to_le_bytes(),
                );
            }
        }
        let projective_z = BigUint::one();
        let bytes = [&SampleEcPoints[0].0, &SampleEcPoints[0].1, &projective_z]
            .into_iter()
            .flat_map(|coordinate| biguint_to_limbs_vec(coordinate, NUM_LIMBS_32).into_iter())
            .collect::<Vec<_>>();
        for byte_offset in (0..3 * NUM_LIMBS_32).step_by(MEMORY_BLOCK_BYTES) {
            unsafe {
                tester.memory.memory.data.write_bytes::<MEMORY_BLOCK_BYTES>(
                    MEMORY_AS,
                    input_pointer + byte_offset as u32,
                    bytes[byte_offset..byte_offset + MEMORY_BLOCK_BYTES]
                        .try_into()
                        .unwrap(),
                );
            }
        }
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(opcode_base + WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize),
            [
                rd_register,
                input_register,
                0,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.preflight,
            &instruction,
            0,
        );
        let execution = harness.preflight.executions.last().unwrap();
        let memory_config = MemoryConfig::default();
        let postflight =
            Postflight::new_for_test(&execution.program, &execution.history, &memory_config)
                .unwrap();
        generate_double_trace_from_postflight(&harness.chip, &postflight, opcode_base).unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_ec_double_cuda_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
        b: BigUint,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness =
            create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());

        // Run some operations
        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                &a,
                &b,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(SampleEcPoints[1].0.clone()),
            Some(SampleEcPoints[1].1.clone()),
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS>(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            &a,
            &b,
            false,
            offset,
            Some(p1_x),
            Some(p1_y),
        );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_ec_double_cuda_2x32() {
        // secp256k1: a=0, b=7
        run_ec_double_cuda_test::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_ec_double_cuda_2x32_nonzero_a_1() {
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        run_ec_double_cuda_test::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
            b,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn test_ec_double_cuda_6x16() {
        // BLS12-381: a=0, b=4
        run_ec_double_cuda_test::<ECC_BLOCKS_48, NUM_LIMBS_48>(
            WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
            BigUint::from(4u32), // BLS12-381 b coefficient,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_preflight_ec_double<const BLOCKS: usize, const NUM_LIMBS: usize>(
        modulus: BigUint,
        a: BigUint,
        b: BigUint,
        is_setup: bool,
        rows: usize,
    ) {
        let offset = WeierstrassOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let mut tester = GpuChipTestBuilder::default();
        let harness = create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone(), b.clone());
        let values = if is_setup {
            vec![modulus, a, b]
        } else {
            // Exercise device input reduction with the smallest noncanonical value and the
            // largest value representable by the declared limb width.
            let one = BigUint::from(1u8);
            let max_value = (&one << (NUM_LIMBS * LIMB_BITS)) - &one;
            assert_ne!(&max_value % &modulus, BigUint::zero());
            vec![modulus + &one, max_value, BigUint::one()]
        };
        let input_bytes = encode_field_inputs(&values, NUM_LIMBS);
        let output_bytes =
            field_expression_output(harness.executor.program(), &input_bytes, is_setup);
        let rs_ptrs = [16u32];
        let rd_ptr = 8u32;
        let rs_vals = [0x100u32];
        let rd_val = 0x300u32;
        initialize_vec_heap_memory::<1, BLOCKS>(
            &mut tester,
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
        );
        let local_opcode = if is_setup {
            WeierstrassOpcode::SETUP_SW_EC_DOUBLE_PROJ
        } else {
            WeierstrassOpcode::SW_EC_DOUBLE_PROJ
        };
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + local_opcode as usize),
            [
                rd_ptr as usize,
                rs_ptrs[0] as usize,
                0,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let device_ctx = tester.range_checker().device_ctx.clone();
        let (_, history) = make_vec_heap_history::<1, BLOCKS>(
            instruction.clone(),
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
            &output_bytes,
        );
        let (program, mut history) = repeat_vec_heap_history(instruction, history, rows);
        let valid_history = history.clone();
        tester.record_preflight_history(&program, &valid_history, Some(0));
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        let replay_ctx = harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
            .unwrap();
        let replay_counts = gpu_range_counts(&tester);

        let write_start = 2 + BLOCKS;
        history.memory.accesses[write_start].value[0] ^= 1;
        let (corrupt_transcript, corrupt_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        assert!(harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &corrupt_transcript, &corrupt_plan)
            .is_err());
        assert_eq!(replay_counts, gpu_range_counts(&tester));

        let mut tester = tester.build();
        tester.balance_preflight_history(&program, &valid_history, Some(0));
        tester
            .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
            .finalize()
            .simple_test()
            .expect("Weierstrass double postflight proof failed");
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_double_preflight_rows_counts_setup_and_corruption_32_48() {
        for is_setup in [false, true] {
            run_preflight_ec_double::<ECC_BLOCKS_32, NUM_LIMBS_32>(
                secp256k1_coord_prime(),
                BigUint::zero(),
                BigUint::from(7u8),
                is_setup,
                1,
            );
            run_preflight_ec_double::<ECC_BLOCKS_48, NUM_LIMBS_48>(
                BLS12_381_MODULUS.clone(),
                BigUint::zero(),
                BigUint::from(4u8),
                is_setup,
                1,
            );
        }
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_double_preflight_fills_finalized_padding_row() {
        run_preflight_ec_double::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            secp256k1_coord_prime(),
            BigUint::zero(),
            BigUint::from(7u8),
            false,
            3,
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // SANITY TESTS
    //
    // Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////

    /// Helper to convert projective (X, Y, Z) to affine (x, y) via x = X/Z, y = Y/Z.
    fn proj_to_affine(
        x_proj: &BigUint,
        y_proj: &BigUint,
        z_proj: &BigUint,
        p: &BigUint,
    ) -> (BigUint, BigUint) {
        // Compute z^{-1} mod p using Fermat's little theorem: z^{-1} = z^{p-2} mod p.
        let z_inv = z_proj.modpow(&(p - BigUint::from(2u32)), p);
        let x_affine = (x_proj * &z_inv) % p;
        let y_affine = (y_proj * &z_inv) % p;
        (x_affine, y_affine)
    }

    #[test]
    fn ec_double_sanity_test_sample_ec_points() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256k1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };

        // secp256k1: a=0, b=7
        let executor = get_ec_double_executor::<{ ECC_BLOCKS_32 }>(
            config,
            tester.range_checker().bus().range_max_bits,
            WeierstrassOpcode::CLASS_OFFSET,
            BigUint::zero(),
            BigUint::from(7u32), // secp256k1 b coefficient,
        );

        let (p1_x, p1_y) = SampleEcPoints[1].clone();

        // Projective input: (X, Y, Z) where Z=1 for affine point.
        let z1 = BigUint::one();
        let vars = executor.program().execute(&[p1_x, p1_y, z1], &[true]);
        // Output vars (X3, Y3, Z3) are the final three variables.
        let r = &vars[vars.len() - 3..];

        // Output is projective coordinates in (X3, Y3, Z3) order.
        assert_eq!(r.len(), 3);

        // Convert projective output to affine and compare.
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, SampleEcPoints[3].0);
        assert_eq!(y3_affine, SampleEcPoints[3].1);
    }

    #[test]
    fn ec_double_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let p = secp256r1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: p.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        // secp256r1: a=-3 (p-3),
        // b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
        let a = BigUint::from_str_radix(
            "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
            16,
        )
        .unwrap();
        // b coefficient (functions compute b3 = 3*b internally)
        let b = BigUint::from_str_radix(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap();

        let executor = get_ec_double_executor::<{ ECC_BLOCKS_32 }>(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            WeierstrassOpcode::CLASS_OFFSET,
            a.clone(),
            b,
        );

        // Testing data from: http://point-at-infinity.org/ecc/nisttv
        let p1_x = BigUint::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .unwrap();
        let p1_y = BigUint::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .unwrap();

        // Projective input: (X, Y, Z) where Z=1 for affine point.
        let z1 = BigUint::one();
        let vars = executor.program().execute(&[p1_x, p1_y, z1], &[true]);
        // Output vars (X3, Y3, Z3) are the final three variables.
        let r = &vars[vars.len() - 3..];

        // Output is projective coordinates in (X3, Y3, Z3) order.
        assert_eq!(r.len(), 3);

        let expected_double_x = BigUint::from_str_radix(
            "7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
            16,
        )
        .unwrap();
        let expected_double_y = BigUint::from_str_radix(
            "07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1",
            16,
        )
        .unwrap();

        // Convert projective output to affine and compare.
        let (x3_affine, y3_affine) = proj_to_affine(&r[0], &r[1], &r[2], &p);
        assert_eq!(x3_affine, expected_double_x);
        assert_eq!(y3_affine, expected_double_y);
    }
}
