use std::array;

use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{solve_write_data, LoadStoreCoreChip, Rv32LoadStoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        Rv32LoadStoreAdapter,
    },
    program::Instruction,
};

const RV32_NUM_CELLS: usize = 4;
const IMM_BITS: usize = 12;
const ADDR_BITS: usize = 29;

type F = BabyBear;

fn num_into_limbs(num: u32) -> [F; 4] {
    array::from_fn(|i| F::from_canonical_u32((num >> (8 * i)) & 255))
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32LoadStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32LoadStoreOpcode,
    is_load: bool,
) {
    let imm: i32 = rng.gen_range(0..(1 << IMM_BITS)) - (1 << (IMM_BITS - 1));
    let imm: i32 = (imm >> 2) << 2;
    let ptr = if imm < 0 {
        rng.gen_range(-imm..(1 << ADDR_BITS))
    } else {
        rng.gen_range(0..((1 << ADDR_BITS) - imm))
    };
    let ptr = ((ptr >> 2) << 2) as u32;
    let a = gen_pointer(rng, 32);
    let b = gen_pointer(rng, 32);
    let ptr_val = (ptr as i32 + imm) as usize;
    let ptr_limbs = num_into_limbs(ptr);

    tester.write(1, b, ptr_limbs);

    let some_prev_data: [F; RV32_NUM_CELLS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    if is_load {
        tester.write(1, a, some_prev_data);
    } else {
        tester.write(2, ptr_val, some_prev_data);
    }

    let data: [F; RV32_NUM_CELLS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    if is_load {
        tester.write(2, ptr_val, data);
    } else {
        tester.write(1, a, data);
    }

    tester.execute(
        chip,
        Instruction::from_isize(
            opcode as usize + Rv32LoadStoreOpcode::default_offset(),
            a as isize,
            b as isize,
            imm as isize,
            1,
            2,
        ),
    );

    let write_data = solve_write_data(opcode, data, some_prev_data);
    if is_load {
        assert_eq!(write_data, tester.read::<4>(1, a));
    } else {
        assert_eq!(write_data, tester.read::<4>(2, ptr_val));
    }
}

#[test]
fn simple_execute_roundtrip_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32LoadStoreAdapter::<F, RV32_NUM_CELLS>::new(
        tester.memory_chip().borrow().range_checker.clone(),
        Rv32LoadStoreOpcode::default_offset(),
    );
    let inner = LoadStoreCoreChip::<F, RV32_NUM_CELLS>::new(adapter.offset);
    let mut chip = Rv32LoadStoreChip::<F>::new(adapter, inner, tester.memory_chip());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADW, true);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREW, false);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREH, false);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREB, false);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADH, true);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADB, true);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADHU, true);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADBU, true);
    }
}

#[test]
fn solve_loadw_storew_sanity_test() {
    let read_data = [138, 45, 202, 76].map(F::from_canonical_u32);
    let prev_data = [159, 213, 89, 34].map(F::from_canonical_u32);
    let store_write_data = solve_write_data(STOREW, read_data, prev_data);
    let load_write_data = solve_write_data(LOADW, read_data, prev_data);
    assert_eq!(store_write_data, read_data);
    assert_eq!(load_write_data, read_data);
}

#[test]
fn solve_storeh_sanity_test() {
    let read_data = [250, 123, 67, 198].map(F::from_canonical_u32);
    let prev_data = [144, 56, 175, 92].map(F::from_canonical_u32);
    let write_data = solve_write_data(STOREH, read_data, prev_data);
    assert_eq!(write_data, [250, 123, 175, 92].map(F::from_canonical_u32));
}

#[test]
fn solve_storeb_sanity_test() {
    let read_data = [221, 104, 58, 147].map(F::from_canonical_u32);
    let prev_data = [199, 83, 243, 12].map(F::from_canonical_u32);
    let write_data = solve_write_data(STOREB, read_data, prev_data);
    assert_eq!(write_data, [221, 83, 243, 12].map(F::from_canonical_u32));
}

#[test]
fn solve_loadhu_sanity_test() {
    let read_data = [175, 33, 198, 250].map(F::from_canonical_u32);
    let prev_data = [90, 121, 64, 205].map(F::from_canonical_u32);
    let write_data = solve_write_data(LOADHU, read_data, prev_data);
    assert_eq!(write_data, [175, 33, 0, 0].map(F::from_canonical_u32));
}

#[test]
fn solve_loadbu_sanity_test() {
    let read_data = [131, 74, 186, 29].map(F::from_canonical_u32);
    let prev_data = [142, 67, 210, 88].map(F::from_canonical_u32);
    let write_data = solve_write_data(LOADBU, read_data, prev_data);
    assert_eq!(write_data, [131, 0, 0, 0].map(F::from_canonical_u32));
}

#[test]
fn solve_loadh_sanity_test() {
    let read_data = [34, 159, 237, 112].map(F::from_canonical_u32);
    let prev_data = [94, 183, 56, 241].map(F::from_canonical_u32);
    let write_data = solve_write_data(LOADH, read_data, prev_data);
    assert_eq!(write_data, [34, 159, 255, 255].map(F::from_canonical_u32));
}

#[test]
fn solve_loadb_sanity_test() {
    let read_data = [103, 151, 78, 219].map(F::from_canonical_u32);
    let prev_data = [53, 180, 29, 244].map(F::from_canonical_u32);
    let write_data = solve_write_data(LOADB, read_data, prev_data);
    assert_eq!(write_data, [103, 0, 0, 0].map(F::from_canonical_u32));
}
