use std::sync::Arc;

use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    interaction::{BusIndex, LookupBus, PermutationCheckBus},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    prover::AirProvingContext,
    test_utils::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    AirRef, StarkEngine,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use super::*;
use crate::{
    arch::{vm_poseidon2_config, PublicValuesState, U16_CELLS_PER_PUBLIC_VALUE},
    system::poseidon2::{new_poseidon2_periphery_air, Poseidon2PeripheryChip},
    utils::test_cpu_engine,
};

type TestSC = BabyBearPoseidon2Config;
type F = BabyBear;

const PUBLIC_VALUES_BUS: BusIndex = 0;
const COMPRESSION_BUS: BusIndex = 1;

struct TestCase {
    airs: Vec<AirRef<TestSC>>,
    contexts: Vec<AirProvingContext<CpuBackend<TestSC>>>,
}

fn test_case(
    state: &PublicValuesState,
    initial_len: usize,
    extra_reveals: &[(usize, u64)],
) -> TestCase {
    let public_values_bus = PublicValuesBus::new(PUBLIC_VALUES_BUS);
    let air = PublicValuesAir::new(
        state.max_public_values() * U16_CELLS_PER_PUBLIC_VALUE,
        public_values_bus,
        PermutationCheckBus::new(COMPRESSION_BUS),
    );
    let hasher = Arc::new(Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3));
    let chip = PublicValuesChip::new(air.clone(), hasher.clone());
    let public_values_ctx = chip.generate_proving_ctx(state, initial_len);

    let mut reveals = state.values()[initial_len..]
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    reveals.extend_from_slice(extra_reveals);

    let mut airs: Vec<AirRef<TestSC>> = vec![Arc::new(air)];
    let mut contexts = vec![public_values_ctx];
    if !reveals.is_empty() {
        let dummy_air = DummyInteractionAir::new(1 + PUBLIC_VALUE_LIMBS, true, PUBLIC_VALUES_BUS);
        let mut values = Vec::new();
        for (ordinal, value) in reveals {
            values.push(F::ONE);
            values.push(F::from_usize(ordinal));
            values.extend(value_limbs::<F>(value));
        }
        let width = 2 + PUBLIC_VALUE_LIMBS;
        let height = (values.len() / width).next_power_of_two();
        values.resize(height * width, F::ZERO);
        airs.push(Arc::new(dummy_air));
        contexts.push(AirProvingContext::simple_no_pis(RowMajorMatrix::new(
            values, width,
        )));
    }

    if state.len() > initial_len {
        airs.push(new_poseidon2_periphery_air(
            vm_poseidon2_config(),
            LookupBus::new(COMPRESSION_BUS),
            3,
        ));
        contexts.push(hasher.generate_proving_ctx());
    }
    TestCase { airs, contexts }
}

fn run(case: TestCase) -> bool {
    test_cpu_engine().run_test(case.airs, case.contexts).is_ok()
}

#[test]
fn valid_initial_and_continued_segments() {
    let mut first = PublicValuesState::new(4);
    first.try_push(0).unwrap();
    first.try_push(0x8877_6655_4433_2211).unwrap();
    assert!(run(test_case(&first, 0, &[])));

    let mut second = first.clone();
    second.try_push(7).unwrap();
    assert!(run(test_case(&second, first.len(), &[])));
    assert!(run(test_case(&second, second.len(), &[])));
}

#[test]
fn duplicate_or_wrong_ordinal_is_rejected() {
    let mut state = PublicValuesState::new(2);
    state.try_push(9).unwrap();
    assert!(!run(test_case(&state, 0, &[(0, 9)])));
    assert!(!run(test_case(&state, 0, &[(1, 9)])));
}

#[test]
fn public_endpoint_mutation_is_rejected() {
    let mut state = PublicValuesState::new(2);
    state.try_push(11).unwrap();
    let mut case = test_case(&state, 0, &[]);
    case.contexts[0].public_values[VM_DIGEST_WIDTH] += F::ONE;
    assert!(!run(case));
}
