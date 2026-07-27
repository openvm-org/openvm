#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{str::FromStr, sync::atomic::Ordering};

use halo2curves_axiom::secp256r1;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num, Zero};
use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, MatrixRecordArena, MemoryConfig, Postflight, PreflightExecutor, PreflightHistory,
    PreflightProgramEvent, TraceFiller, MEMORY_BLOCK_BYTES,
};
use openvm_circuit_primitives::bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime};
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::generate_random_biguint, utils::biguint_to_limbs_vec, ExprBuilderConfig,
};
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_stark_backend::{p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::extension::HybridWeierstrassChip,
    openvm_circuit::arch::testing::{
        default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    openvm_circuit::arch::rvr::{cuda::GpuPostflightProgram, PreflightEndpoint, PreflightEventLog},
    openvm_circuit::system::cuda::memory::MemoryInventoryGPU,
    openvm_circuit::{
        arch::{DenseRecordArena, VirtualMachine, VmExecutor},
        utils::{test_gpu_engine, test_system_config},
    },
    openvm_circuit_primitives::Chip,
    openvm_cuda_backend::{base::DeviceMatrix, GpuBackend},
    openvm_cuda_common::copy::MemCopyD2H,
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        SystemOpcode,
    },
    openvm_mod_circuit_builder::{run_field_expression_precomputed, FieldExpressionProgram},
    openvm_stark_backend::{prover::AirProvingContext, StarkEngine},
    rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PREFLIGHT_WRITE_BIT},
    strum::EnumCount,
};

use crate::{
    get_ec_addne_air, get_ec_addne_chip, get_ec_addne_executor, get_ec_double_air,
    get_ec_double_chip, get_ec_double_executor,
    weierstrass_chip::{
        generate_add_ne_trace_from_postflight, generate_double_trace_from_postflight,
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
fn make_vec_heap_transcript<const NUM_READS: usize, const BLOCKS: usize>(
    instruction: Instruction<F>,
    rs_ptrs: [u32; NUM_READS],
    rd_ptr: u32,
    rs_vals: [u32; NUM_READS],
    rd_val: u32,
    input_bytes: &[u8],
    output_bytes: &[u8],
) -> (Program<F>, PreflightEventLog) {
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
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: register / 2,
            value: register_block(pointer),
        });
        timestamp += 1;
    }
    memory_log.push(PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: RV64_REGISTER_AS,
        pointer: rd_ptr / 2,
        value: register_block(rd_val),
    });
    timestamp += 1;
    for (read, &pointer) in rs_vals.iter().enumerate() {
        for block in 0..BLOCKS {
            let start = read * bytes_per_value + block * MEMORY_BLOCK_BYTES;
            memory_log.push(PreflightMemoryEvent {
                timestamp,
                address_space_and_kind: RV64_MEMORY_AS,
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
            address_space_and_kind: RV64_MEMORY_AS | PREFLIGHT_WRITE_BIT,
            pointer,
            value: packed_u16_block(&output_bytes[start..start + MEMORY_BLOCK_BYTES]),
        });
        initial_write_log.push(PreflightInitialWrite {
            address_space: RV64_MEMORY_AS,
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
        PreflightEventLog {
            program_log: vec![
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
            memory_log,
            initial_write_log,
        },
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn repeat_vec_heap_transcript(
    instruction: Instruction<F>,
    transcript: PreflightEventLog,
    repetitions: usize,
) -> (Program<F>, PreflightEventLog) {
    assert!(repetitions > 0);
    let first_timestamp = transcript.program_log[0].timestamp;
    let timestamp_step = transcript.program_log[1].timestamp - first_timestamp;
    let mut memory_log = Vec::with_capacity(transcript.memory_log.len() * repetitions);
    let mut program_log = Vec::with_capacity(repetitions + 2);
    for repetition in 0..repetitions {
        let timestamp_shift = repetition as u32 * timestamp_step;
        program_log.push(PreflightProgramEvent {
            pc: repetition as u32 * 4,
            timestamp: first_timestamp + timestamp_shift,
        });
        memory_log.extend(transcript.memory_log.iter().copied().map(|mut event| {
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
        PreflightEventLog {
            program_log,
            memory_log,
            initial_write_log: transcript.initial_write_log,
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
                RV64_REGISTER_AS,
                register / 2,
                [pointer as u16, (pointer >> 16) as u16, 0, 0],
            );
        }
    }
    unsafe {
        tester.memory.memory.data.write::<u16, 4>(
            RV64_REGISTER_AS,
            rd_ptr / 2,
            [rd_val as u16, (rd_val >> 16) as u16, 0, 0],
        );
    }
    for (read, &pointer) in rs_vals.iter().enumerate() {
        for block in 0..BLOCKS {
            let start = read * bytes_per_value + block * MEMORY_BLOCK_BYTES;
            unsafe {
                tester.memory.memory.data.write::<u16, 4>(
                    RV64_MEMORY_AS,
                    pointer / 2 + (block * 4) as u32,
                    packed_u16_block(&input_bytes[start..start + MEMORY_BLOCK_BYTES]),
                );
            }
        }
    }
    for block in 0..BLOCKS {
        unsafe {
            tester.memory.memory.data.write::<u16, 4>(
                RV64_MEMORY_AS,
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
fn combine_two_vec_heap_transcripts(
    first_instruction: Instruction<F>,
    mut first: PreflightEventLog,
    second_instruction: Instruction<F>,
    mut second: PreflightEventLog,
) -> (Program<F>, PreflightEventLog) {
    let second_start = first.program_log[1].timestamp;
    let timestamp_shift = second_start - second.program_log[0].timestamp;
    for event in &mut second.memory_log {
        event.timestamp += timestamp_shift;
    }
    let second_end = second.program_log[1].timestamp + timestamp_shift;
    first.memory_log.extend(second.memory_log);
    first.initial_write_log.extend(second.initial_write_log);
    first.program_log = vec![
        first.program_log[0],
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

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct NonWeierstrassChip;

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl Chip<DenseRecordArena, GpuBackend> for NonWeierstrassChip {
    fn generate_proving_ctx(&self, _arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        AirProvingContext::simple_no_pis(DeviceMatrix::dummy())
    }
}

mod ec_addne_tests {
    use num_traits::One;

    use super::*;
    use crate::EcAddNeExecutor;

    type EcAddneHarness<const BLOCKS: usize> = TestChipHarness<
        F,
        EcAddNeExecutor<BLOCKS>,
        WeierstrassAir<2, BLOCKS>,
        WeierstrassChip<F, 2, BLOCKS>,
    >;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> EcAddneHarness<BLOCKS> {
        let air = get_ec_addne_air::<BLOCKS>(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
        );
        let executor = get_ec_addne_executor::<BLOCKS>(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            tester.address_bits(),
            offset,
        );
        let chip = get_ec_addne_chip::<F, BLOCKS>(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
        );

        EcAddneHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
    }

    #[cfg(feature = "cuda")]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        EcAddNeExecutor<BLOCKS>,
        WeierstrassAir<2, BLOCKS>,
        HybridWeierstrassChip<F, 2, BLOCKS>,
        WeierstrassChip<F, 2, BLOCKS>,
    >;

    #[cfg(feature = "cuda")]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_ec_addne_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
        );
        let executor = get_ec_addne_executor(
            config.clone(),
            range_bus.range_max_bits,
            tester.address_bits(),
            offset,
        );

        let cpu_chip = get_ec_addne_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
        );

        let gpu_cpu_chip = get_ec_addne_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
        );
        #[cfg(feature = "rvr")]
        let hybrid_chip = HybridWeierstrassChip::new_with_replay(
            gpu_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            offset,
            tester.range_checker(),
        );
        #[cfg(not(feature = "rvr"))]
        let hybrid_chip =
            HybridWeierstrassChip::new(gpu_cpu_chip, tester.range_checker().device_ctx.clone());

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_addne<const BLOCKS: usize, const NUM_LIMBS: usize, RA: Arena>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcAddNeExecutor<BLOCKS>,
        arena: &mut RA,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
        p1: Option<(BigUint, BigUint)>,
        p2: Option<(BigUint, BigUint)>,
    ) -> Instruction<F>
    where
        EcAddNeExecutor<BLOCKS>: PreflightExecutor<F, RA>,
    {
        let (x1, y1, x2, y2, op_local) = if is_setup {
            (
                modulus.clone(),
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
                Rv64WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            )
        } else if let Some((x1, y1)) = p1 {
            let (x2, y2) = p2.unwrap();
            let x1 = x1 % modulus;
            let y1 = y1 % modulus;
            let x2 = x2 % modulus;
            let y2 = y2 % modulus;
            if rng.random_bool(0.5) {
                (x1, y1, x2, y2, Rv64WeierstrassOpcode::EC_ADD_NE as usize)
            } else {
                (x2, y2, x1, y1, Rv64WeierstrassOpcode::EC_ADD_NE as usize)
            }
        } else {
            panic!("Generating random inputs generically is harder because the input points need to be on the curve.");
        };

        let ptr_as = RV64_REGISTER_AS as usize;
        let data_as = RV64_MEMORY_AS as usize;

        let rs1_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
        let rs2_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
        let rd_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);

        let p1_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let p2_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let result_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;

        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs2_ptr,
            p2_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
            ptr_as,
            rd_ptr,
            result_base_addr.to_le_bytes().map(F::from_u8),
        );

        let x1_limbs: Vec<F> = biguint_to_limbs_vec(&x1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let x2_limbs: Vec<F> = biguint_to_limbs_vec(&x2, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let y1_limbs: Vec<F> = biguint_to_limbs_vec(&y1, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let y2_limbs: Vec<F> = biguint_to_limbs_vec(&y2, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

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
                p2_base_addr as usize + i,
                x2_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );

            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                (p2_base_addr + NUM_LIMBS as u64) as usize + i,
                y2_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
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

        tester.execute(executor, arena, &instruction);
        instruction
    }

    fn run_ec_addne_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset);

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            true,
            offset,
            None,
            None,
        );

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            false,
            offset,
            Some(SampleEcPoints[0].clone()),
            Some(SampleEcPoints[1].clone()),
        );

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
        );

        let tester = tester.build().load(harness).finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_ec_addne_32limb() {
        run_ec_addne_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
        );
    }

    #[test]
    fn test_ec_addne_48limb() {
        run_ec_addne_test::<{ ECC_BLOCKS_48 }, { NUM_LIMBS_48 }>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
        );
    }

    #[test]
    fn ec_addne_postflight_matches_legacy_and_rejects_bad_pointer() {
        let mut tester = VmChipTestBuilder::<F>::default();
        let modulus = secp256k1_coord_prime();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let opcode_base = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let mut harness = create_harness::<ECC_BLOCKS_32>(&tester, config, opcode_base);

        let rd_register = 24usize;
        let lhs_register = 8usize;
        let rhs_register = 16usize;
        let rd_pointer = 0x300u32;
        let lhs_pointer = 0x100u32;
        let rhs_pointer = 0x200u32;
        for (register, pointer) in [
            (rd_register, rd_pointer),
            (lhs_register, lhs_pointer),
            (rhs_register, rhs_pointer),
        ] {
            unsafe {
                tester.memory.memory.data.write_bytes(
                    RV64_REGISTER_AS,
                    register as u32,
                    u64::from(pointer).to_le_bytes(),
                );
            }
        }
        for (pointer, (x, y)) in [
            (lhs_pointer, &SampleEcPoints[0]),
            (rhs_pointer, &SampleEcPoints[1]),
        ] {
            let bytes = [x, y]
                .into_iter()
                .flat_map(|coordinate| biguint_to_limbs_vec(coordinate, NUM_LIMBS_32).into_iter())
                .collect::<Vec<_>>();
            for byte_offset in (0..2 * NUM_LIMBS_32).step_by(MEMORY_BLOCK_BYTES) {
                unsafe {
                    tester.memory.memory.data.write_bytes::<MEMORY_BLOCK_BYTES>(
                        RV64_MEMORY_AS,
                        pointer + byte_offset as u32,
                        bytes[byte_offset..byte_offset + MEMORY_BLOCK_BYTES]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
        }
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(opcode_base + Rv64WeierstrassOpcode::EC_ADD_NE as usize),
            [
                rd_register,
                lhs_register,
                rhs_register,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        tester.execute_with_pc(&mut harness.executor, &mut harness.arena, &instruction, 0);
        let final_timestamp = tester.memory.memory.timestamp();
        let mut history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: final_timestamp,
                },
            ],
            memory: tester.memory.memory.take_log(),
        };
        let program =
            Program::new_without_debug_infos(&[instruction.clone(), instruction.clone()], 0);
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let actual =
            generate_add_ne_trace_from_postflight(&harness.chip, &postflight, opcode_base).unwrap();

        let rows_used = harness.arena.trace_offset / harness.arena.width;
        let mut expected_values = harness.arena.trace_buffer.clone();
        expected_values.truncate(rows_used.next_power_of_two() * harness.arena.width);
        let mut expected = RowMajorMatrix::new(expected_values, harness.arena.width);
        harness.chip.inner.fill_trace(
            &harness.chip.mem_helper.as_borrowed(),
            &mut expected,
            rows_used,
        );
        assert_eq!(actual, expected);

        drop(postflight);
        history.memory.accesses[0].value[2] = 1;
        let malformed = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let error = generate_add_ne_trace_from_postflight(&harness.chip, &malformed, opcode_base)
            .unwrap_err();
        assert!(
            error.to_string().contains("nonzero upper 32 bits"),
            "{error}"
        );

        history.memory.accesses[0].value[2] = 0;
        history.memory.accesses[0].pointer += 1;
        let error = Postflight::new(&program, &history, &memory_config, None)
            .err()
            .expect("misaligned memory event must be rejected");
        assert!(error.to_string().contains("misaligned"), "{error}");

        history.memory.accesses[0].pointer -= 1;
        let write = history
            .memory
            .accesses
            .iter_mut()
            .find(|event| event.is_write())
            .unwrap();
        write.value[0] ^= 1;
        let range_counts_before = harness
            .chip
            .inner
            .range_checker
            .count
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        let malformed = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let error = generate_add_ne_trace_from_postflight(&harness.chip, &malformed, opcode_base)
            .unwrap_err();
        assert!(error.to_string().contains("OutputMismatch"), "{error}");
        let range_counts_after = harness
            .chip
            .inner
            .range_checker
            .count
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        assert_eq!(range_counts_after, range_counts_before);
    }

    #[cfg(feature = "cuda")]
    fn run_cuda_ec_addne<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
    ) {
        use crate::EccRecord;

        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset);

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            true,
            offset,
            None,
            None,
        );

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            false,
            offset,
            Some(SampleEcPoints[0].clone()),
            Some(SampleEcPoints[1].clone()),
        );

        set_and_execute_ec_addne::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            false,
            offset,
            Some(SampleEcPoints[2].clone()),
            Some(SampleEcPoints[3].clone()),
        );

        harness
            .dense_arena
            .get_record_seeker::<EccRecord<2, BLOCKS>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                harness.executor.get_record_layout::<F>(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_weierstrass_addne_cuda_2x32() {
        run_cuda_ec_addne::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_weierstrass_addne_cuda_6x16() {
        run_cuda_ec_addne::<ECC_BLOCKS_48, NUM_LIMBS_48>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_preflight_ec_addne<const BLOCKS: usize, const NUM_LIMBS: usize>(
        modulus: BigUint,
        is_setup: bool,
    ) {
        let offset = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset);
        let values = if is_setup {
            vec![
                modulus,
                BigUint::from(1u8),
                BigUint::from(1u8),
                BigUint::from(1u8),
            ]
        } else {
            vec![
                BigUint::from(1u8),
                BigUint::from(2u8),
                BigUint::from(3u8),
                BigUint::from(4u8),
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
            Rv64WeierstrassOpcode::SETUP_EC_ADD_NE
        } else {
            Rv64WeierstrassOpcode::EC_ADD_NE
        };
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + local_opcode as usize),
            [
                rd_ptr as usize,
                rs_ptrs[0] as usize,
                rs_ptrs[1] as usize,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        // This isolated chip proof still needs the harness's memory lookup
        // counterpart. The chip trace itself remains the replay-derived context below.
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
            0,
        );
        let device_ctx = tester.range_checker().device_ctx.clone();
        let (program, mut transcript) = make_vec_heap_transcript::<2, BLOCKS>(
            instruction,
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
            &output_bytes,
        );
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        let replay_ctx = harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
            .unwrap();
        let replay_counts = gpu_range_counts(&tester);

        let write_start = 3 + 2 * BLOCKS;
        transcript.memory_log[write_start].value[0] ^= 1;
        let (corrupt_transcript, corrupt_plan) = gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        assert!(harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &corrupt_transcript, &corrupt_plan)
            .is_err());
        assert_eq!(replay_counts, gpu_range_counts(&tester));

        tester
            .build()
            .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
            .finalize()
            .simple_test()
            .expect("Weierstrass add checkpoint replay proof failed");
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_add_preflight_rows_counts_setup_and_corruption_32_48() {
        for is_setup in [false, true] {
            run_preflight_ec_addne::<ECC_BLOCKS_32, NUM_LIMBS_32>(
                secp256k1_coord_prime(),
                is_setup,
            );
            run_preflight_ec_addne::<ECC_BLOCKS_48, NUM_LIMBS_48>(
                BLS12_381_MODULUS.clone(),
                is_setup,
            );
        }
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_coordinator_distinguishes_repeated_curve_instances() {
        let first_base = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let second_base = first_base + Rv64WeierstrassOpcode::COUNT;
        let config = ExprBuilderConfig {
            modulus: secp256k1_coord_prime(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let tester = GpuChipTestBuilder::default();
        let first = create_cuda_harness::<ECC_BLOCKS_32>(&tester, config.clone(), first_base);
        let second = create_cuda_harness::<ECC_BLOCKS_32>(&tester, config, second_base);

        let first_input = encode_field_inputs(
            &[
                BigUint::from(1u8),
                BigUint::from(2u8),
                BigUint::from(3u8),
                BigUint::from(4u8),
            ],
            NUM_LIMBS_32,
        );
        let second_input = encode_field_inputs(
            &[
                BigUint::from(5u8),
                BigUint::from(6u8),
                BigUint::from(7u8),
                BigUint::from(8u8),
            ],
            NUM_LIMBS_32,
        );
        let first_output = field_expression_output(first.executor.program(), &first_input, false);
        let second_output =
            field_expression_output(second.executor.program(), &second_input, false);
        let first_instruction = Instruction::from_usize(
            VmOpcode::from_usize(first_base + Rv64WeierstrassOpcode::EC_ADD_NE as usize),
            [
                8,
                16,
                24,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        let second_instruction = Instruction::from_usize(
            VmOpcode::from_usize(second_base + Rv64WeierstrassOpcode::EC_ADD_NE as usize),
            [
                32,
                40,
                48,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        let (_, first_transcript) = make_vec_heap_transcript::<2, ECC_BLOCKS_32>(
            first_instruction.clone(),
            [16, 24],
            8,
            [0x100, 0x200],
            0x300,
            &first_input,
            &first_output,
        );
        let (_, second_transcript) = make_vec_heap_transcript::<2, ECC_BLOCKS_32>(
            second_instruction.clone(),
            [40, 48],
            32,
            [0x400, 0x500],
            0x600,
            &second_input,
            &second_output,
        );
        let (program, transcript) = combine_two_vec_heap_transcripts(
            first_instruction,
            first_transcript,
            second_instruction,
            second_transcript,
        );
        let device_ctx = tester.range_checker().device_ctx.clone();
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
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
        assert!(incomplete
            .generate_for_chip(&NonWeierstrassChip)
            .unwrap()
            .is_none());
        drop(
            incomplete
                .generate_for_chip(&first.gpu_chip)
                .unwrap()
                .unwrap(),
        );
        let error = incomplete
            .finish()
            .err()
            .expect("the second curve's opcode must remain unclaimed");
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
        // on the same expanded transcript. Both curve instances have the same concrete chip types;
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
                    .map(|(offset, byte)| ((RV64_REGISTER_AS, register + offset as u32), byte)),
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
                    .map(|(offset, byte)| ((RV64_MEMORY_AS, pointer + offset as u32), byte)),
            );
        }
        let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
        let mut vm_config = crate::Rv64WeierstrassConfig::new(vec![
            crate::SECP256K1_CONFIG.clone(),
            crate::SECP256K1_CONFIG.clone(),
        ]);
        *vm_config.as_mut() = test_system_config();
        let executor = VmExecutor::new(vm_config.clone()).unwrap();
        let state = executor
            .interpreter_instance(&exe)
            .unwrap()
            .create_initial_vm_state(Vec::<Vec<u8>>::new());

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
        let poisoned_gpu_program = GpuPostflightProgram::upload(
            &program,
            &incomplete_config.modular.system.memory_config,
            &poisoned_vm.engine.device().device_ctx,
        )
        .unwrap();
        let (poisoned_transcript, poisoned_plan) = poisoned_gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        let late_coverage_error = crate::WeierstrassPreflightGpuTracegen::new(
            &extension,
            &poisoned_gpu_program,
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
            &poisoned_gpu_program,
            &poisoned_transcript,
            &poisoned_plan,
        )
        .generate_proving_ctx(&mut poisoned_vm, &incomplete_config.modular.modular, None)
        .err()
        .expect("a failed preflight tracegen session must poison retries");
        assert!(retry_error.to_string().contains("poisoned"));

        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            crate::Rv64WeierstrassHybridBuilder,
            vm_config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let vm_gpu_program = GpuPostflightProgram::upload(
            &program,
            &vm_config.modular.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (vm_transcript, vm_plan) = vm_gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        let proving_ctx = crate::WeierstrassPreflightGpuTracegen::new(
            &vm_config.weierstrass,
            &vm_gpu_program,
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
        let (retry_transcript, retry_plan) = vm_gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        let retry_ctx = crate::WeierstrassPreflightGpuTracegen::new(
            &vm_config.weierstrass,
            &vm_gpu_program,
            &retry_transcript,
            &retry_plan,
        )
        .generate_proving_ctx(&mut vm, &vm_config.modular.modular, None)
        .expect("successful outer coordination must permit another preflight tracegen session");
        drop(retry_ctx);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /// SANITY TESTS
    ///
    /// Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////
    #[test]
    fn ec_addne_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: secp256k1_coord_prime(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };

        let executor = get_ec_addne_executor::<{ ECC_BLOCKS_32 }>(
            config,
            tester.range_checker().bus().range_max_bits,
            tester.address_bits(),
            Rv64WeierstrassOpcode::CLASS_OFFSET,
        );

        let (p1_x, p1_y) = SampleEcPoints[0].clone();
        let (p2_x, p2_y) = SampleEcPoints[1].clone();
        assert_eq!(executor.program().num_vars(), 3); // lambda, x3, y3
        let r = executor
            .program()
            .execute(&[p1_x, p1_y, p2_x, p2_y], &[true]);

        assert_eq!(r.len(), 3); // lambda, x3, y3
        assert_eq!(r[1], SampleEcPoints[2].0);
        assert_eq!(r[2], SampleEcPoints[2].1);

        let (p1_x, p1_y) = SampleEcPoints[2].clone();
        let (p2_x, p2_y) = SampleEcPoints[3].clone();
        assert_eq!(executor.program().num_vars(), 3); // lambda, x3, y3
        let r = executor
            .program()
            .execute(&[p1_x, p1_y, p2_x, p2_y], &[true]);

        assert_eq!(r.len(), 3); // lambda, x3, y3
        assert_eq!(r[1], SampleEcPoints[4].0);
        assert_eq!(r[2], SampleEcPoints[4].1);
    }
}

mod ec_double_tests {
    use super::*;

    type EcDoubleHarness<const BLOCKS: usize> = TestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS>,
        WeierstrassAir<1, BLOCKS>,
        WeierstrassChip<F, 1, BLOCKS>,
        MatrixRecordArena<F>,
    >;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
    ) -> EcDoubleHarness<BLOCKS> {
        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
            a_biguint.clone(),
        );
        let executor = get_ec_double_executor(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
        );
        let chip = get_ec_double_chip(
            config.clone(),
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
            a_biguint,
        );
        EcDoubleHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
    }

    #[cfg(feature = "cuda")]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        EcDoubleExecutor<BLOCKS>,
        WeierstrassAir<1, BLOCKS>,
        HybridWeierstrassChip<F, 1, BLOCKS>,
        WeierstrassChip<F, 1, BLOCKS>,
    >;

    #[cfg(feature = "cuda")]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
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
        );
        let executor = get_ec_double_executor(
            config.clone(),
            range_bus.range_max_bits,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
        );

        let cpu_chip = get_ec_double_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
            a_biguint.clone(),
        );
        let gpu_cpu_chip = get_ec_double_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
            a_biguint,
        );
        #[cfg(feature = "rvr")]
        let hybrid_chip = HybridWeierstrassChip::new_with_replay(
            gpu_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            offset,
            tester.range_checker(),
        );
        #[cfg(not(feature = "rvr"))]
        let hybrid_chip =
            HybridWeierstrassChip::new(gpu_cpu_chip, tester.range_checker().device_ctx.clone());

        GpuTestChipHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_ec_double<const BLOCKS: usize, const NUM_LIMBS: usize, RA: Arena>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut EcDoubleExecutor<BLOCKS>,
        arena: &mut RA,
        rng: &mut StdRng,
        modulus: &BigUint,
        a_biguint: &BigUint,
        is_setup: bool,
        offset: usize,
        x: Option<BigUint>,
        y: Option<BigUint>,
    ) -> Instruction<F>
    where
        EcDoubleExecutor<BLOCKS>: PreflightExecutor<F, RA>,
    {
        let (x1, y1, op_local) = if is_setup {
            (
                modulus.clone(),
                a_biguint.clone(),
                Rv64WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
            )
        } else if let Some(x) = x {
            let y = y.unwrap();
            let x = x % modulus;
            let y = y % modulus;
            (x, y, Rv64WeierstrassOpcode::EC_DOUBLE as usize)
        } else {
            let x = generate_random_biguint(modulus);
            let y = generate_random_biguint(modulus);

            (x, y, Rv64WeierstrassOpcode::EC_DOUBLE as usize)
        };

        let ptr_as = RV64_REGISTER_AS as usize;
        let data_as = RV64_MEMORY_AS as usize;

        let rs1_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
        let rd_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);

        let p1_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;
        let result_base_addr = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u64;

        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_u8),
        );
        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(
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
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            0,
            ptr_as as isize,
            data_as as isize,
        );

        tester.execute(executor, arena, &instruction);
        instruction
    }

    fn run_ec_double_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset, a.clone());

        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
                &mut rng,
                &modulus,
                &a,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
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

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &modulus,
            &a,
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
        run_ec_double_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
        );
    }

    #[test]
    fn test_ec_double_32limb_nonzero_a() {
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);

        run_ec_double_test::<{ ECC_BLOCKS_32 }, { NUM_LIMBS_32 }>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
        );
    }

    #[test]
    fn test_ec_double_48limb() {
        run_ec_double_test::<{ ECC_BLOCKS_48 }, { NUM_LIMBS_48 }>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
        );
    }

    #[test]
    fn ec_double_postflight_matches_legacy() {
        let mut tester = VmChipTestBuilder::<F>::default();
        let modulus = secp256k1_coord_prime();
        let a = BigUint::zero();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let opcode_base = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let mut harness = create_harness::<ECC_BLOCKS_32>(&tester, config, opcode_base, a.clone());

        let rd_register = 16usize;
        let input_register = 8usize;
        let rd_pointer = 0x200u32;
        let input_pointer = 0x100u32;
        for (register, pointer) in [(rd_register, rd_pointer), (input_register, input_pointer)] {
            unsafe {
                tester.memory.memory.data.write_bytes(
                    RV64_REGISTER_AS,
                    register as u32,
                    u64::from(pointer).to_le_bytes(),
                );
            }
        }
        let bytes = [&SampleEcPoints[0].0, &SampleEcPoints[0].1]
            .into_iter()
            .flat_map(|coordinate| biguint_to_limbs_vec(coordinate, NUM_LIMBS_32).into_iter())
            .collect::<Vec<_>>();
        for byte_offset in (0..2 * NUM_LIMBS_32).step_by(MEMORY_BLOCK_BYTES) {
            unsafe {
                tester.memory.memory.data.write_bytes::<MEMORY_BLOCK_BYTES>(
                    RV64_MEMORY_AS,
                    input_pointer + byte_offset as u32,
                    bytes[byte_offset..byte_offset + MEMORY_BLOCK_BYTES]
                        .try_into()
                        .unwrap(),
                );
            }
        }
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(opcode_base + Rv64WeierstrassOpcode::EC_DOUBLE as usize),
            [
                rd_register,
                input_register,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        tester.execute_with_pc(&mut harness.executor, &mut harness.arena, &instruction, 0);
        let final_timestamp = tester.memory.memory.timestamp();
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: final_timestamp,
                },
            ],
            memory: tester.memory.memory.take_log(),
        };
        let program =
            Program::new_without_debug_infos(&[instruction.clone(), instruction.clone()], 0);
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let actual =
            generate_double_trace_from_postflight(&harness.chip, &postflight, opcode_base).unwrap();

        let rows_used = harness.arena.trace_offset / harness.arena.width;
        let mut expected_values = harness.arena.trace_buffer.clone();
        expected_values.truncate(rows_used.next_power_of_two() * harness.arena.width);
        let mut expected = RowMajorMatrix::new(expected_values, harness.arena.width);
        harness.chip.inner.fill_trace(
            &harness.chip.mem_helper.as_borrowed(),
            &mut expected,
            rows_used,
        );
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "cuda")]
    fn run_ec_double_cuda_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        offset: usize,
        modulus: BigUint,
        num_ops: usize,
        a: BigUint,
    ) {
        use crate::EccRecord;

        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone());

        // Run some operations
        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                &modulus,
                &a,
                i == 0,
                offset,
                None,
                None,
            );
        }

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            false,
            offset,
            Some(SampleEcPoints[0].0.clone()),
            Some(SampleEcPoints[0].1.clone()),
        );

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
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

        set_and_execute_ec_double::<BLOCKS, NUM_LIMBS, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &modulus,
            &a,
            false,
            offset,
            Some(p1_x),
            Some(p1_y),
        );

        harness
            .dense_arena
            .get_record_seeker::<EccRecord<1, BLOCKS>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                harness.executor.get_record_layout::<F>(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_ec_double_cuda_2x32() {
        run_ec_double_cuda_test::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256k1_coord_prime(),
            50,
            BigUint::zero(),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_ec_double_cuda_2x32_nonzero_a_1() {
        let coeff_a = (-secp256r1::Fp::from(3)).to_bytes();
        let a = BigUint::from_bytes_le(&coeff_a);

        run_ec_double_cuda_test::<ECC_BLOCKS_32, NUM_LIMBS_32>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            secp256r1_coord_prime(),
            50,
            a,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_ec_double_cuda_6x16() {
        run_ec_double_cuda_test::<ECC_BLOCKS_48, NUM_LIMBS_48>(
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            BLS12_381_MODULUS.clone(),
            50,
            BigUint::zero(),
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_preflight_ec_double<const BLOCKS: usize, const NUM_LIMBS: usize>(
        modulus: BigUint,
        a: BigUint,
        is_setup: bool,
        rows: usize,
    ) {
        let offset = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset, a.clone());
        let values = if is_setup {
            vec![modulus, a]
        } else {
            // Exercise device input reduction with the smallest noncanonical value and the
            // largest value representable by the declared limb width.
            let one = BigUint::from(1u8);
            let max_value = (&one << (NUM_LIMBS * LIMB_BITS)) - &one;
            assert_ne!(&max_value % &modulus, BigUint::zero());
            vec![modulus + &one, max_value]
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
            Rv64WeierstrassOpcode::SETUP_EC_DOUBLE
        } else {
            Rv64WeierstrassOpcode::EC_DOUBLE
        };
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + local_opcode as usize),
            [
                rd_ptr as usize,
                rs_ptrs[0] as usize,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        );
        // This isolated chip proof still needs the harness's memory lookup
        // counterpart. The chip trace itself remains the replay-derived context below.
        for row in 0..rows {
            tester.execute_with_pc(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
                row as u32 * 4,
            );
        }
        let device_ctx = tester.range_checker().device_ctx.clone();
        let (_, transcript) = make_vec_heap_transcript::<1, BLOCKS>(
            instruction.clone(),
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            &input_bytes,
            &output_bytes,
        );
        let (program, mut transcript) = repeat_vec_heap_transcript(instruction, transcript, rows);
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &openvm_circuit::arch::MemoryConfig::default(),
            &device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        let replay_ctx = harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
            .unwrap();
        let replay_counts = gpu_range_counts(&tester);

        let write_start = 2 + BLOCKS;
        transcript.memory_log[write_start].value[0] ^= 1;
        let (corrupt_transcript, corrupt_plan) = gpu_program
            .upload_transcript(&transcript, PreflightEndpoint::Terminated)
            .unwrap();
        assert!(harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &corrupt_transcript, &corrupt_plan)
            .is_err());
        assert_eq!(replay_counts, gpu_range_counts(&tester));

        tester
            .build()
            .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
            .finalize()
            .simple_test()
            .expect("Weierstrass double checkpoint replay proof failed");
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn weierstrass_double_preflight_rows_counts_setup_and_corruption_32_48() {
        for is_setup in [false, true] {
            run_preflight_ec_double::<ECC_BLOCKS_32, NUM_LIMBS_32>(
                secp256k1_coord_prime(),
                BigUint::zero(),
                is_setup,
                1,
            );
            run_preflight_ec_double::<ECC_BLOCKS_48, NUM_LIMBS_48>(
                BLS12_381_MODULUS.clone(),
                BigUint::zero(),
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
            false,
            3,
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /// SANITY TESTS
    ///
    /// Ensure that execute functions produce the correct results.
    ///////////////////////////////////////////////////////////////////////////////////////
    #[test]
    fn ec_double_sanity_test_sample_ec_points() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: secp256k1_coord_prime(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };

        let executor = get_ec_double_executor::<{ ECC_BLOCKS_32 }>(
            config,
            tester.range_checker().bus().range_max_bits,
            tester.address_bits(),
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            BigUint::zero(),
        );

        let (p1_x, p1_y) = SampleEcPoints[1].clone();

        assert_eq!(executor.program().num_vars(), 3); // lambda, x3, y3

        let r = executor.program().execute(&[p1_x, p1_y], &[true]);
        assert_eq!(r.len(), 3); // lambda, x3, y3
        assert_eq!(r[1], SampleEcPoints[3].0);
        assert_eq!(r[2], SampleEcPoints[3].1);
    }

    #[test]
    fn ec_double_sanity_test() {
        let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: secp256r1_coord_prime(),
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let a = BigUint::from_str_radix(
            "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
            16,
        )
        .unwrap();

        let executor = get_ec_double_executor::<{ ECC_BLOCKS_32 }>(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            tester.address_bits(),
            Rv64WeierstrassOpcode::CLASS_OFFSET,
            a.clone(),
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

        assert_eq!(executor.program().num_vars(), 3); // lambda, x3, y3

        let r = executor.program().execute(&[p1_x, p1_y], &[true]);
        assert_eq!(r.len(), 3); // lambda, x3, y3
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
        assert_eq!(r[1], expected_double_x);
        assert_eq!(r[2], expected_double_y);
    }
}
