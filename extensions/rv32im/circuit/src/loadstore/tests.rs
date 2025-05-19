use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, VmChipTestBuilder},
        VmAirWrapper,
    },
    utils::u32_into_limbs,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::FieldAlgebra,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use test_case::test_case;

use super::{run_write_data, LoadStoreCoreAir, LoadStoreStep, Rv32LoadStoreChip};
use crate::{
    adapters::{
        compose, Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterCols, Rv32LoadStoreAdapterStep,
        RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    loadstore::LoadStoreCoreCols,
    test_utils::get_verification_error,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;

fn create_test_chip(tester: &mut VmChipTestBuilder<F>) -> Rv32LoadStoreChip<F> {
    let range_checker_chip = tester.range_checker();
    

    Rv32LoadStoreChip::<F>::new(
        VmAirWrapper::new(
            Rv32LoadStoreAdapterAir::new(
                tester.memory_bridge(),
                tester.execution_bridge(),
                range_checker_chip.bus(),
                tester.address_bits(),
            ),
            LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
        ),
        LoadStoreStep::new(
            Rv32LoadStoreAdapterStep::new(tester.address_bits()),
            range_checker_chip.clone(),
            Rv32LoadStoreOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    )
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32LoadStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32LoadStoreOpcode,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    mem_as: Option<usize>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or(rng.gen_range(0..2));
    let imm_ext = imm + imm_sign * (0xffffffff ^ ((1 << IMM_BITS) - 1));

    let alignment = match opcode {
        LOADW | STOREW => 2,
        LOADHU | STOREH => 1,
        LOADBU | STOREB => 0,
        _ => unreachable!(),
    };

    let ptr_val = rng.gen_range(
        0..(1 << (tester.memory_controller().mem_config().pointer_max_bits - alignment)),
    ) << alignment;

    let rs1 = rs1
        .unwrap_or(u32_into_limbs::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            (ptr_val as u32).wrapping_sub(imm_ext),
        ))
        .map(F::from_canonical_u32);
    let a = gen_pointer(rng, 4);
    let b = gen_pointer(rng, 4);
    let is_load = [LOADW, LOADHU, LOADBU].contains(&opcode);
    let mem_as = mem_as.unwrap_or(if is_load {
        *[1, 2].choose(rng).unwrap()
    } else {
        *[2, 3, 4].choose(rng).unwrap()
    });

    let ptr_val = imm_ext.wrapping_add(compose(rs1));
    let shift_amount = ptr_val % 4;
    tester.write(1, b, rs1);

    let mut some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
    let mut read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));

    if is_load {
        if a == 0 {
            some_prev_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        }
        tester.write(1, a, some_prev_data);
        if mem_as == 1 && ptr_val - shift_amount == 0 {
            read_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        }
        tester.write(mem_as, (ptr_val - shift_amount) as usize, read_data);
    } else {
        if a == 0 {
            read_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        }
        tester.write(mem_as, (ptr_val - shift_amount) as usize, some_prev_data);
        tester.write(1, a, read_data);
    }

    let enabled_write = !(is_load & (a == 0));

    tester.execute(
        chip,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                1,
                mem_as,
                enabled_write as usize,
                imm_sign as usize,
            ],
        ),
    );

    let write_data = run_write_data(opcode, read_data, some_prev_data, shift_amount);
    if is_load {
        if enabled_write {
            assert_eq!(write_data, tester.read::<4>(1, a));
        } else {
            assert_eq!([F::ZERO; RV32_REGISTER_NUM_LIMBS], tester.read::<4>(1, a));
        }
    } else {
        assert_eq!(
            write_data,
            tester.read::<4>(mem_as, (ptr_val - shift_amount) as usize)
        );
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test_case(LOADW, 100)]
#[test_case(LOADBU, 100)]
#[test_case(LOADHU, 100)]
#[test_case(STOREW, 100)]
#[test_case(STOREB, 100)]
#[test_case(STOREH, 100)]
fn rand_loadstore_test(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode,
            None,
            None,
            None,
            None,
        );
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct LoadStorePrankValues {
    read_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    prev_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    write_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    flags: Option<[u32; 4]>,
    is_load: Option<bool>,
    mem_as: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_loadstore_test(
    opcode: Rv32LoadStoreOpcode,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    prank_vals: LoadStorePrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        rs1,
        imm,
        imm_sign,
        None,
    );

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut LoadStoreCoreCols<F, RV32_REGISTER_NUM_LIMBS> = core_row.borrow_mut();

        if let Some(read_data) = prank_vals.read_data {
            core_cols.read_data = read_data.map(F::from_canonical_u32);
        }
        if let Some(prev_data) = prank_vals.prev_data {
            core_cols.prev_data = prev_data.map(F::from_canonical_u32);
        }
        if let Some(write_data) = prank_vals.write_data {
            core_cols.write_data = write_data.map(F::from_canonical_u32);
        }
        if let Some(flags) = prank_vals.flags {
            core_cols.flags = flags.map(F::from_canonical_u32);
        }
        if let Some(is_load) = prank_vals.is_load {
            core_cols.is_load = F::from_bool(is_load);
        }
        if let Some(mem_as) = prank_vals.mem_as {
            adapter_cols.mem_as = F::from_canonical_u32(mem_as);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn negative_wrong_opcode_tests() {
    run_negative_loadstore_test(
        LOADW,
        None,
        None,
        None,
        LoadStorePrankValues {
            is_load: Some(false),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        LOADBU,
        Some([4, 0, 0, 0]),
        Some(1),
        None,
        LoadStorePrankValues {
            flags: Some([0, 0, 0, 2]),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STOREH,
        Some([11, 169, 76, 28]),
        Some(37121),
        None,
        LoadStorePrankValues {
            flags: Some([1, 0, 1, 0]),
            is_load: Some(true),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn negative_write_data_tests() {
    run_negative_loadstore_test(
        LOADHU,
        Some([13, 11, 156, 23]),
        Some(43641),
        None,
        LoadStorePrankValues {
            read_data: Some([175, 33, 198, 250]),
            prev_data: Some([90, 121, 64, 205]),
            write_data: Some([175, 33, 0, 0]),
            flags: Some([0, 2, 0, 0]),
            is_load: Some(true),
            mem_as: None,
        },
        true,
    );

    run_negative_loadstore_test(
        STOREB,
        Some([45, 123, 87, 24]),
        Some(28122),
        Some(0),
        LoadStorePrankValues {
            read_data: Some([175, 33, 198, 250]),
            prev_data: Some([90, 121, 64, 205]),
            write_data: Some([175, 121, 64, 205]),
            flags: Some([0, 0, 1, 1]),
            is_load: None,
            mem_as: None,
        },
        false,
    );
}

#[test]
fn negative_wrong_address_space_tests() {
    run_negative_loadstore_test(
        LOADW,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(3),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        LOADW,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(4),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STOREW,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(1),
            ..Default::default()
        },
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn run_loadw_storew_sanity_test() {
    let read_data = [138, 45, 202, 76].map(F::from_canonical_u32);
    let prev_data = [159, 213, 89, 34].map(F::from_canonical_u32);
    let store_write_data = run_write_data(STOREW, read_data, prev_data, 0);
    let load_write_data = run_write_data(LOADW, read_data, prev_data, 0);
    assert_eq!(store_write_data, read_data);
    assert_eq!(load_write_data, read_data);
}

#[test]
fn run_storeh_sanity_test() {
    let read_data = [250, 123, 67, 198].map(F::from_canonical_u32);
    let prev_data = [144, 56, 175, 92].map(F::from_canonical_u32);
    let write_data = run_write_data(STOREH, read_data, prev_data, 0);
    let write_data2 = run_write_data(STOREH, read_data, prev_data, 2);
    assert_eq!(write_data, [250, 123, 175, 92].map(F::from_canonical_u32));
    assert_eq!(write_data2, [144, 56, 250, 123].map(F::from_canonical_u32));
}

#[test]
fn run_storeb_sanity_test() {
    let read_data = [221, 104, 58, 147].map(F::from_canonical_u32);
    let prev_data = [199, 83, 243, 12].map(F::from_canonical_u32);
    let write_data = run_write_data(STOREB, read_data, prev_data, 0);
    let write_data1 = run_write_data(STOREB, read_data, prev_data, 1);
    let write_data2 = run_write_data(STOREB, read_data, prev_data, 2);
    let write_data3 = run_write_data(STOREB, read_data, prev_data, 3);
    assert_eq!(write_data, [221, 83, 243, 12].map(F::from_canonical_u32));
    assert_eq!(write_data1, [199, 221, 243, 12].map(F::from_canonical_u32));
    assert_eq!(write_data2, [199, 83, 221, 12].map(F::from_canonical_u32));
    assert_eq!(write_data3, [199, 83, 243, 221].map(F::from_canonical_u32));
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = [175, 33, 198, 250].map(F::from_canonical_u32);
    let prev_data = [90, 121, 64, 205].map(F::from_canonical_u32);
    let write_data = run_write_data(LOADHU, read_data, prev_data, 0);
    let write_data2 = run_write_data(LOADHU, read_data, prev_data, 2);
    assert_eq!(write_data, [175, 33, 0, 0].map(F::from_canonical_u32));
    assert_eq!(write_data2, [198, 250, 0, 0].map(F::from_canonical_u32));
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = [131, 74, 186, 29].map(F::from_canonical_u32);
    let prev_data = [142, 67, 210, 88].map(F::from_canonical_u32);
    let write_data = run_write_data(LOADBU, read_data, prev_data, 0);
    let write_data1 = run_write_data(LOADBU, read_data, prev_data, 1);
    let write_data2 = run_write_data(LOADBU, read_data, prev_data, 2);
    let write_data3 = run_write_data(LOADBU, read_data, prev_data, 3);
    assert_eq!(write_data, [131, 0, 0, 0].map(F::from_canonical_u32));
    assert_eq!(write_data1, [74, 0, 0, 0].map(F::from_canonical_u32));
    assert_eq!(write_data2, [186, 0, 0, 0].map(F::from_canonical_u32));
    assert_eq!(write_data3, [29, 0, 0, 0].map(F::from_canonical_u32));
}
