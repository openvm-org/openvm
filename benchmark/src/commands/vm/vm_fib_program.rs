use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

use super::benchmark_helpers::vm_benchmark_execute_and_prove;

pub fn benchmark_fib_program(n: usize) -> Result<()> {
    println!("Running Fibonacci program benchmark with n = {}", n);

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Felt<_> = builder.eval(F::zero());
    let b: Felt<_> = builder.eval(F::one());
    let n_ext: Var<_> = builder.eval(F::from_canonical_usize(n));

    let start: Var<_> = builder.eval(F::zero());
    let end = n_ext;

    builder.range(start, end).for_each(|_, builder| {
        let temp: Felt<_> = builder.uninit();
        builder.assign(&temp, b);
        builder.assign(&b, a + b);
        builder.assign(&a, temp);
    });

    builder.halt();

    let fib_program = builder.compile_isa::<1>();

    vm_benchmark_execute_and_prove::<1>(fib_program, vec![], "VM Fibonacci Program")
}
