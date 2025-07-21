use openvm_circuit::{arch::PUBLIC_VALUES_AIR_ID, utils::air_test_impl};
use openvm_native_circuit::{execute_program_with_config, test_native_config, NativeCpuBuilder};
use openvm_native_compiler::{asm::AsmBuilder, prelude::*};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    p3_baby_bear::BabyBear,
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_public_values() {
    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);
    let mut builder = AsmBuilder::<F, EF>::default();

    {
        let a: Felt<_> = builder.constant(public_value_0);
        let b: Felt<_> = builder.constant(public_value_1);

        let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
        let var_array = builder.dyn_array::<Felt<_>>(dyn_len);
        builder.set(&var_array, RVar::zero(), a);
        builder.set(&var_array, RVar::one(), b);

        builder.commit_public_values(&var_array);

        builder.halt();
    }

    let program = builder.compile_isa();
    let mut config = test_native_config();
    config.system.num_public_values = 2;
    // This is to justify using log_blowup=1
    assert!(config.as_ref().max_constraint_degree <= 3);
    let fri_params = FriParameters::new_for_testing(1);
    let (_, mut vdata) = air_test_impl::<BabyBearPoseidon2Engine, _>(
        fri_params,
        NativeCpuBuilder,
        config,
        program,
        vec![],
        1,
        true,
    )
    .unwrap();
    assert_eq!(vdata.len(), 1);
    let proof = vdata.pop().unwrap().data.proof;
    assert_eq!(
        &proof.get_public_values()[PUBLIC_VALUES_AIR_ID],
        &[public_value_0, public_value_1]
    );
}

#[test]
fn test_compiler_public_values_no_initial() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let public_value_0 = F::from_canonical_u32(10);
    let public_value_1 = F::from_canonical_u32(20);

    let a: Felt<_> = builder.constant(public_value_0);
    let b: Felt<_> = builder.constant(public_value_1);

    let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(2));
    let var_array = builder.dyn_array::<Felt<_>>(dyn_len);
    builder.set(&var_array, RVar::zero(), a);
    builder.set(&var_array, RVar::one(), b);

    builder.commit_public_values(&var_array);

    builder.halt();

    let program = builder.compile_isa();
    let (output, _) = execute_program_with_config::<BabyBearPoseidon2Engine, _>(
        program,
        vec![],
        NativeCpuBuilder,
        test_native_config(),
    )
    .unwrap();
    assert_eq!(output.system_records.public_values[0], public_value_0);
    assert_eq!(output.system_records.public_values[1], public_value_1);
}
