use openvm_circuit::arch::{testing::VmChipTestBuilder, VmAirWrapper};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::NativeLoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{NativeLoadStoreChip, NativeLoadStoreCoreAir, NativeLoadStoreCoreStep};
use crate::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterStep,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

#[derive(Debug)]
struct TestData {
    a: F,
    b: F,
    c: F,
    d: F,
    e: F,
    ad_val: F,
    cd_val: F,
    data_val: F,
    is_load: bool,
    is_hint: bool,
}

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> NativeLoadStoreChip<F, 1> {
    let mut chip = NativeLoadStoreChip::<F, 1>::new(
        VmAirWrapper::new(
            NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            NativeLoadStoreCoreAir::new(NativeLoadStoreOpcode::CLASS_OFFSET),
        ),
        NativeLoadStoreCoreStep::new(
            NativeLoadStoreAdapterStep::new(NativeLoadStoreOpcode::CLASS_OFFSET),
            NativeLoadStoreOpcode::CLASS_OFFSET,
        ),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    chip
}

fn gen_test_data(rng: &mut StdRng, opcode: NativeLoadStoreOpcode) -> TestData {
    let is_load = matches!(opcode, NativeLoadStoreOpcode::LOADW);

    let a = rng.gen_range(0..1 << 20);
    let b = rng.gen_range(0..1 << 20);
    let c = rng.gen_range(0..1 << 20);
    let d = F::from_canonical_u32(4u32);
    let e = F::from_canonical_u32(4u32);

    TestData {
        a: F::from_canonical_u32(a),
        b: F::from_canonical_u32(b),
        c: F::from_canonical_u32(c),
        d,
        e,
        ad_val: F::from_canonical_u32(111),
        cd_val: F::from_canonical_u32(222),
        data_val: F::from_canonical_u32(444),
        is_load,
        is_hint: matches!(opcode, NativeLoadStoreOpcode::HINT_STOREW),
    }
}

fn get_data_pointer(data: &TestData) -> F {
    if data.d != F::ZERO {
        data.cd_val + data.b
    } else {
        data.c + data.b
    }
}

fn set_values(tester: &mut VmChipTestBuilder<F>, data: &TestData) {
    if data.d != F::ZERO {
        tester.write(
            data.d.as_canonical_u32() as usize,
            data.a.as_canonical_u32() as usize,
            [data.ad_val],
        );
        tester.write(
            data.d.as_canonical_u32() as usize,
            data.c.as_canonical_u32() as usize,
            [data.cd_val],
        );
    }
    if data.is_load {
        let data_pointer = get_data_pointer(data);
        tester.write(
            data.e.as_canonical_u32() as usize,
            data_pointer.as_canonical_u32() as usize,
            [data.data_val],
        );
    }
    if data.is_hint {
        for _ in 0..data.e.as_canonical_u32() {
            tester.streams.hint_stream.push_back(data.data_val);
        }
    }
}

fn check_values(tester: &mut VmChipTestBuilder<F>, data: &TestData) {
    let data_pointer = get_data_pointer(data);

    let written_data_val = if data.is_load {
        tester.read::<1>(
            data.d.as_canonical_u32() as usize,
            data.a.as_canonical_u32() as usize,
        )[0]
    } else {
        tester.read::<1>(
            data.e.as_canonical_u32() as usize,
            data_pointer.as_canonical_u32() as usize,
        )[0]
    };

    let correct_data_val = if data.is_load || data.is_hint {
        data.data_val
    } else if data.d != F::ZERO {
        data.ad_val
    } else {
        data.a
    };

    assert_eq!(written_data_val, correct_data_val, "{:?}", data);
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut NativeLoadStoreChip<F, 1>,
    rng: &mut StdRng,
    opcode: NativeLoadStoreOpcode,
) {
    let data = gen_test_data(rng, opcode);
    set_values(tester, &data);

    tester.execute_with_pc(
        chip,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [data.a, data.b, data.c, data.d, data.e].map(|x| x.as_canonical_u32() as usize),
        ),
        0u32,
    );

    check_values(tester, &data);
}

#[test]
fn rand_native_loadstore_test() {
    setup_tracing();

    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    for _ in 0..20 {
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREW);
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADW);
    }
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
