use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, seq::IndexedRandom, Rng};
use test_case::test_case;

use super::{
    aligned::LoadStoreAlignedCoreCols,
    byte::{LoadStoreByteCoreAir, LoadStoreByteCoreCols, LoadStoreByteFiller},
    common::run_write_data,
    doubleword::{
        LoadStoreDoublewordCoreAir, LoadStoreDoublewordCoreCols, LoadStoreDoublewordFiller,
    },
    halfword::{LoadStoreHalfwordCoreAir, LoadStoreHalfwordFiller, HALFWORD_SELECTOR_WIDTH},
    word::{LoadStoreWordCoreAir, LoadStoreWordFiller, WORD_SELECTOR_WIDTH},
    Rv64LoadStoreByteAir, Rv64LoadStoreByteChip, Rv64LoadStoreByteExecutor,
    Rv64LoadStoreDoublewordAir, Rv64LoadStoreDoublewordChip, Rv64LoadStoreDoublewordExecutor,
    Rv64LoadStoreHalfwordAir, Rv64LoadStoreHalfwordChip, Rv64LoadStoreHalfwordExecutor,
    Rv64LoadStoreWordAir, Rv64LoadStoreWordChip, Rv64LoadStoreWordExecutor,
};
use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16,
    Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor, Rv64LoadStoreAdapterFiller,
    RV64_BYTE_BITS,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

type ByteHarness =
    TestChipHarness<F, Rv64LoadStoreByteExecutor, Rv64LoadStoreByteAir, Rv64LoadStoreByteChip<F>>;
type HalfwordHarness = TestChipHarness<
    F,
    Rv64LoadStoreHalfwordExecutor,
    Rv64LoadStoreHalfwordAir,
    Rv64LoadStoreHalfwordChip<F>,
>;
type WordHarness =
    TestChipHarness<F, Rv64LoadStoreWordExecutor, Rv64LoadStoreWordAir, Rv64LoadStoreWordChip<F>>;
type DoublewordHarness = TestChipHarness<
    F,
    Rv64LoadStoreDoublewordExecutor,
    Rv64LoadStoreDoublewordAir,
    Rv64LoadStoreDoublewordChip<F>,
>;

fn u16_block_to_f_bytes(block: [u16; BLOCK_FE_WIDTH]) -> [F; 8] {
    rv64_u16_block_to_bytes(block).map(F::from_u8)
}

fn create_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    ByteHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let air = Rv64LoadStoreByteAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadStoreByteExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreByteChip::<F>::new(
        LoadStoreByteFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        ByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

fn create_halfword_harness(tester: &mut VmChipTestBuilder<F>) -> HalfwordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreHalfwordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreHalfwordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreHalfwordChip::<F>::new(
        LoadStoreHalfwordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_word_harness(tester: &mut VmChipTestBuilder<F>) -> WordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreWordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreWordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreWordChip::<F>::new(
        LoadStoreWordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    WordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_doubleword_harness(tester: &mut VmChipTestBuilder<F>) -> DoublewordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreDoublewordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreDoublewordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreDoublewordChip::<F>::new(
        LoadStoreDoublewordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    DoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut VmChipTestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
) {
    set_and_execute_with(tester, executor, arena, rng, opcode, None, None, None, None);
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute_with<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut VmChipTestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
    rs1: Option<[u8; 8]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    mem_as: Option<usize>,
) {
    let imm = imm.unwrap_or_else(|| rng.random_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or_else(|| rng.random_range(0..2));
    let imm_ext = sign_extend_imm16(imm, imm_sign);
    let alignment = match opcode {
        LOADD | STORED => 3,
        LOADW | LOADWU | STOREW => 2,
        LOADH | LOADHU | STOREH => 1,
        LOADB | LOADBU | STOREB => 0,
    };

    let ptr_val: u32 = rng.random_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    let ptr = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
    let rs1 = rs1.unwrap_or([ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0]);
    let rs1_low = rv64_bytes_to_u32(rs1);
    let ptr_val = imm_ext.wrapping_add(rs1_low);
    let shift_amount = (ptr_val as usize) & 7;
    let base_ptr = (ptr_val as usize) - shift_amount;

    let max_addr = 1usize << tester.address_bits();
    let a = rng.random_range(0..(max_addr - 8)) / 8 * 8;
    let b = rng.random_range(0..(max_addr - 8)) / 8 * 8;
    let is_load = matches!(
        opcode,
        LOADD | LOADW | LOADH | LOADB | LOADWU | LOADHU | LOADBU
    );
    let mem_as: usize = mem_as.unwrap_or_else(|| {
        if is_load {
            2
        } else {
            *[2usize, PUBLIC_VALUES_AS as usize].choose(rng).unwrap()
        }
    });

    tester.write_bytes(RV64_REGISTER_AS as usize, b, rs1.map(F::from_u8));

    let mut prev_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    let mut read_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    if is_load {
        if a == 0 {
            prev_data = [0; BLOCK_FE_WIDTH];
        }
        tester.write_bytes(
            RV64_REGISTER_AS as usize,
            a,
            u16_block_to_f_bytes(prev_data),
        );
        tester.write_bytes(mem_as, base_ptr, u16_block_to_f_bytes(read_data));
    } else {
        if a == 0 {
            read_data = [0; BLOCK_FE_WIDTH];
        }
        tester.write_bytes(mem_as, base_ptr, u16_block_to_f_bytes(prev_data));
        tester.write_bytes(
            RV64_REGISTER_AS as usize,
            a,
            u16_block_to_f_bytes(read_data),
        );
    }

    let enabled_write = !(is_load && a == 0);
    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                RV64_REGISTER_AS as usize,
                mem_as,
                enabled_write as usize,
                imm_sign as usize,
            ],
        ),
    );

    let write_data = run_write_data(opcode, read_data, prev_data, shift_amount);
    if is_load {
        let expected = if enabled_write {
            u16_block_to_f_bytes(write_data)
        } else {
            [F::ZERO; 8]
        };
        assert_eq!(
            expected,
            tester.read_bytes::<8>(RV64_REGISTER_AS as usize, a)
        );
    } else {
        assert_eq!(
            u16_block_to_f_bytes(write_data),
            tester.read_bytes::<8>(mem_as, base_ptr)
        );
    }
}

fn memory_config_for(opcodes: &[Rv64LoadStoreOpcode]) -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    if opcodes
        .iter()
        .any(|opcode| matches!(opcode, STORED | STOREW | STOREH | STOREB))
    {
        mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    }
    mem_config
}

#[test_case(LOADBU, 100)]
#[test_case(STOREB, 100)]
fn rand_loadstore_byte_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
        );
    }
    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn positive_loadwu_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[LOADWU]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute_with(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(2),
    );
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[LOADHU]));
    let mut harness = create_halfword_harness(&mut tester);
    set_and_execute_with(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(2),
    );
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn positive_storew_public_values_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[STOREW]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute_with(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(PUBLIC_VALUES_AS as usize),
    );
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(LOADHU, 100)]
#[test_case(STOREH, 100)]
fn rand_loadstore_halfword_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_halfword_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
        );
    }
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(LOADWU, 100)]
#[test_case(STOREW, 100)]
fn rand_loadstore_word_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_word_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
        );
    }
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(LOADD, 100)]
#[test_case(STORED, 100)]
fn rand_loadstore_doubleword_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_doubleword_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
        );
    }
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

fn b(bytes: [u8; 8]) -> [u16; BLOCK_FE_WIDTH] {
    rv64_bytes_to_u16_block(bytes)
}

#[test]
fn run_loadd_stored_sanity_test() {
    let read_data = b([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = b([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(run_write_data(LOADD, read_data, prev_data, 0), read_data);
    assert_eq!(run_write_data(STORED, read_data, prev_data, 0), read_data);
}

#[test]
fn run_loadwu_storew_sanity_test() {
    let read_data = b([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = b([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 0),
        b([138, 45, 202, 76, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 4),
        b([131, 74, 186, 29, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 0),
        b([138, 45, 202, 76, 142, 67, 210, 88])
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 4),
        b([159, 213, 89, 34, 138, 45, 202, 76])
    );
}

#[test]
fn run_storeh_sanity_test() {
    let read_data = b([250, 123, 67, 198, 175, 33, 198, 250]);
    let prev_data = b([144, 56, 175, 92, 90, 121, 64, 205]);
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 0),
        b([250, 123, 175, 92, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 2),
        b([144, 56, 250, 123, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 4),
        b([144, 56, 175, 92, 250, 123, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 6),
        b([144, 56, 175, 92, 90, 121, 250, 123])
    );
}

#[test]
fn run_storeb_sanity_test() {
    let read_data = b([221, 104, 58, 147, 175, 33, 198, 250]);
    let prev_data = b([199, 83, 243, 12, 90, 121, 64, 205]);
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 0),
        b([221, 83, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 1),
        b([199, 221, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 2),
        b([199, 83, 221, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 3),
        b([199, 83, 243, 221, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 4),
        b([199, 83, 243, 12, 221, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 5),
        b([199, 83, 243, 12, 90, 221, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 6),
        b([199, 83, 243, 12, 90, 121, 221, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 7),
        b([199, 83, 243, 12, 90, 121, 64, 221])
    );
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = b([175, 33, 198, 250, 131, 74, 186, 29]);
    let prev_data = b([90, 121, 64, 205, 142, 67, 210, 88]);
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 0),
        b([175, 33, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 2),
        b([198, 250, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 4),
        b([131, 74, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 6),
        b([186, 29, 0, 0, 0, 0, 0, 0])
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = b([131, 74, 186, 29, 138, 45, 202, 76]);
    let prev_data = b([142, 67, 210, 88, 159, 213, 89, 34]);
    for (shift, expected) in [
        (0, [131, 0, 0, 0, 0, 0, 0, 0]),
        (1, [74, 0, 0, 0, 0, 0, 0, 0]),
        (2, [186, 0, 0, 0, 0, 0, 0, 0]),
        (3, [29, 0, 0, 0, 0, 0, 0, 0]),
        (4, [138, 0, 0, 0, 0, 0, 0, 0]),
        (5, [45, 0, 0, 0, 0, 0, 0, 0]),
        (6, [202, 0, 0, 0, 0, 0, 0, 0]),
        (7, [76, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(
            run_write_data(LOADBU, read_data, prev_data, shift),
            b(expected)
        );
    }
}

#[test]
fn load_sign_extend_sanity_tests() {
    let read_data = b([34, 159, 237, 151, 100, 200, 50, 25]);
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([34, 159, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 2),
        b([237, 151, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([100, 200, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 6),
        b([50, 25, 0, 0, 0, 0, 0, 0])
    );

    let read_data = b([45, 82, 99, 127, 200, 150, 180, 210]);
    for shift in 0..8 {
        let byte = rv64_u16_block_to_bytes(read_data)[shift];
        assert_eq!(
            rv64_u16_block_to_bytes(run_write_data(LOADB, read_data, [0; BLOCK_FE_WIDTH], shift)),
            (byte as i8 as i64).to_le_bytes(),
            "LOADB shift={shift}"
        );
    }

    let read_data = b([0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xFF, 0xFF, 0xFF])
    );

    let read_data = b([0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0x7D]);
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([0xAA, 0xBB, 0xCC, 0x7D, 0, 0, 0, 0])
    );
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_2() {
    run_write_data(LOADW, b([1, 2, 3, 4, 5, 6, 7, 8]), [0; BLOCK_FE_WIDTH], 2);
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_6() {
    run_write_data(LOADW, b([1, 2, 3, 4, 5, 6, 7, 8]), [0; BLOCK_FE_WIDTH], 6);
}

#[test]
fn accepted_shift_sets() {
    let read_data = b([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
    for shift in 0..8 {
        let _ = run_write_data(LOADB, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
    for shift in [0, 2, 4, 6] {
        let _ = run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], shift);
        let _ = run_write_data(LOADHU, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
    for shift in [0, 4] {
        let _ = run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], shift);
        let _ = run_write_data(LOADWU, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
}

fn assert_pranked_byte_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreByteCoreCols<F>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked byte loadstore trace should fail");
}

fn assert_pranked_halfword_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreAlignedCoreCols<F, HALFWORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_halfword_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked halfword loadstore trace should fail");
}

fn assert_pranked_word_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreAlignedCoreCols<F, WORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked word loadstore trace should fail");
}

fn assert_pranked_doubleword_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreDoublewordCoreCols<F>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_doubleword_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked doubleword loadstore trace should fail");
}

#[test]
fn negative_split_write_data_tests() {
    assert_pranked_byte_fails(STOREB, |core| core.read_data[0] += F::ONE);
    assert_pranked_halfword_fails(LOADHU, |core| core.read_data[0] += F::ONE);
    assert_pranked_word_fails(LOADWU, |core| core.read_data[0] += F::ONE);
    assert_pranked_doubleword_fails(LOADD, |core| core.read_data[0] += F::ONE);
}

#[test]
fn negative_split_opcode_role_tests() {
    assert_pranked_byte_fails(LOADBU, |core| core.is_load = F::ZERO);
    assert_pranked_halfword_fails(STOREH, |core| core.is_load = F::ONE);
    assert_pranked_word_fails(LOADWU, |core| core.is_load = F::ZERO);
    assert_pranked_doubleword_fails(LOADD, |core| core.is_load = F::ZERO);
}
