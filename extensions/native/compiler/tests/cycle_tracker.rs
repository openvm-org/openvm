use std::ops::Deref;

use openvm_native_circuit::execute_program;
use openvm_native_compiler::{asm::AsmBuilder, conversion::CompilerOptions, ir::Var};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_cycle_tracker() {
    let mut builder = AsmBuilder::<F, EF>::default();

    builder.cycle_tracker_start("test_unclosed");

    builder.cycle_tracker_start("test_outer");

    let n_val = F::from_canonical_u32(10);
    let m_val = F::from_canonical_u32(20);

    let n: Var<_> = builder.eval(n_val);
    let m: Var<_> = builder.eval(m_val);

    let total: Var<_> = builder.eval(F::ZERO);

    builder.cycle_tracker_start("loop");

    for _ in 0..3 {
        let n_plus_m: Var<_> = builder.eval(n + m);
        builder.assign(&total, total + n_plus_m);
    }

    builder.cycle_tracker_end("loop");

    builder.cycle_tracker_end("test_outer");

    builder.halt();

    // after TERMINATE, so this CT_END opcode will not be executed
    builder.cycle_tracker_end("test_unclosed");

    let program = builder.compile_isa_with_options(CompilerOptions {
        enable_cycle_tracker: true,
        ..Default::default()
    });

    for (i, debug_info) in program.debug_infos().deref().iter().enumerate() {
        println!("debug_info {}: {:?}", i, debug_info);
    }

    println!("{}", program);
    execute_program(program, vec![]);
}
