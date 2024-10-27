use axvm_circuit::system::program::util::execute_program;
use axvm_native_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

fn fibonacci(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        let mut a = 0;
        let mut b = 1;
        for _ in 0..n {
            let temp = b;
            b += a;
            a = temp;
        }
        a
    }
}

fn main() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let n_val = 10;
    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Felt<_> = builder.eval(F::zero());
    let b: Felt<_> = builder.eval(F::one());
    let n: Var<_> = builder.eval(F::from_canonical_u32(n_val));

    let start: Var<_> = builder.eval(F::zero());
    let end = n;

    builder.range(start, end).for_each(|_, builder| {
        let temp: Felt<_> = builder.uninit();
        builder.assign(&temp, b);
        builder.assign(&b, a + b);
        builder.assign(&a, temp);
    });

    let expected_value = F::from_canonical_u32(fibonacci(n_val));
    builder.assert_felt_eq(a, expected_value);

    //builder.print_f(a);
    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}
