use std::sync::Arc;

use openvm_circuit::arch::{testing::VmChipTestBuilder, Streams};
use openvm_instructions::{instruction::Instruction, UsizeOpcode, VmOpcode};
use openvm_native_compiler::NativeLoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::{AbstractField, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng};

use super::{
    super::adapters::loadstore_native_adapter::NativeLoadStoreAdapterChip, NativeLoadStoreChip,
    NativeLoadStoreCoreChip,
};

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

fn setup() -> (StdRng, VmChipTestBuilder<F>, NativeLoadStoreChip<F, 1>) {
    let rng = create_seeded_rng();
    let tester = VmChipTestBuilder::default();

    let adapter = NativeLoadStoreAdapterChip::<F, 1>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        NativeLoadStoreOpcode::default_offset(),
    );
    let mut inner = NativeLoadStoreCoreChip::new(NativeLoadStoreOpcode::default_offset());
    inner.set_streams(Arc::new(Mutex::new(Streams::default())));
    let chip = NativeLoadStoreChip::<F, 1>::new(adapter, inner, tester.memory_controller());
    (rng, tester, chip)
}

fn gen_test_data(rng: &mut StdRng, is_immediate: bool, opcode: NativeLoadStoreOpcode) -> TestData {
    let is_load = matches!(opcode, NativeLoadStoreOpcode::LOADW);

    let a = rng.gen_range(0..1 << 20);
    let b = rng.gen_range(0..1 << 20);
    let c = rng.gen_range(0..1 << 20);
    let d = if is_immediate {
        F::ZERO
    } else {
        F::from_canonical_u32(rng.gen_range(1..4))
    };
    let e = F::from_canonical_u32(rng.gen_range(1..4));

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
        is_hint: matches!(opcode, NativeLoadStoreOpcode::SHINTW),
    }
}

fn get_data_pointer(data: &TestData) -> F {
    if data.d != F::ZERO {
        data.cd_val + data.b
    } else {
        data.c + data.b
    }
}

fn set_values(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut NativeLoadStoreChip<F, 1>,
    data: &TestData,
) {
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
            chip.core
                .streams
                .get()
                .unwrap()
                .lock()
                .hint_stream
                .push_back(data.data_val);
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
    is_immediate: bool,
    opcode: NativeLoadStoreOpcode,
) {
    let data = gen_test_data(rng, is_immediate, opcode);
    set_values(tester, chip, &data);

    tester.execute_with_pc(
        chip,
        Instruction::from_usize(
            VmOpcode::with_default_offset(opcode),
            [data.a, data.b, data.c, data.d, data.e].map(|x| x.as_canonical_u32() as usize),
        ),
        0u32,
    );

    check_values(tester, &data);
}

#[test]
fn rand_native_loadstore_test() {
    let (mut rng, mut tester, mut chip) = setup();
    for _ in 0..20 {
        set_and_execute(&mut tester, &mut chip, &mut rng, false, STOREW);
        set_and_execute(&mut tester, &mut chip, &mut rng, false, SHINTW);
        set_and_execute(&mut tester, &mut chip, &mut rng, false, LOADW);

        set_and_execute(&mut tester, &mut chip, &mut rng, true, STOREW);
        set_and_execute(&mut tester, &mut chip, &mut rng, true, SHINTW);
    }
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
