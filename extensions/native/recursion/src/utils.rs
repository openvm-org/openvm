use openvm_native_compiler::ir::{Builder, CanSelect, Config, Felt, MemVariable, Var};
use openvm_stark_backend::{
    p3_commit::TwoAdicMultiplicativeCoset,
    p3_field::{FieldAlgebra, TwoAdicField},
};
use openvm_stark_sdk::config::FriParameters;

use crate::fri::{types::FriConfigVariable, TwoAdicMultiplicativeCosetVariable};

pub fn const_fri_config<C: Config>(
    builder: &mut Builder<C>,
    params: &FriParameters,
) -> FriConfigVariable<C> {
    let two_adicity = C::F::TWO_ADICITY;
    let generators = builder.array(two_adicity + 1);
    let subgroups = builder.array(two_adicity + 1);
    for i in 0..=C::F::TWO_ADICITY {
        let constant_generator = C::F::two_adic_generator(i);
        builder.set(&generators, i, constant_generator);

        let constant_domain = TwoAdicMultiplicativeCoset {
            log_n: i,
            shift: C::F::ONE,
        };
        let domain_value: TwoAdicMultiplicativeCosetVariable<_> = builder.constant(constant_domain);
        // ATTENTION: here must use `builder.set_value`. `builder.set` will convert `Usize::Const`
        // to `Usize::Var` because it calls `builder.eval`.
        builder.set_value(&subgroups, i, domain_value);
    }
    FriConfigVariable {
        log_blowup: params.log_blowup,
        blowup: 1 << params.log_blowup,
        log_final_poly_len: params.log_final_poly_len,
        num_queries: params.num_queries,
        proof_of_work_bits: params.proof_of_work_bits,
        subgroups,
        generators,
    }
}

/// Reference: <https://github.com/Plonky3/Plonky3/blob/622375885320ac6bf3c338001760ed8f2230e3cb/field/src/helpers.rs#L136>
pub fn reduce_32<C: Config>(builder: &mut Builder<C>, vals: &[Felt<C::F>]) -> Var<C::N> {
    let mut power = C::N::ONE;
    let result: Var<C::N> = builder.eval(C::N::ZERO);
    for val in vals.iter() {
        let val = builder.cast_felt_to_var(*val);
        builder.assign(&result, result + val * power);
        power *= C::N::from_canonical_usize(1usize << 32);
    }
    result
}

/// Reference: <https://github.com/Plonky3/Plonky3/blob/622375885320ac6bf3c338001760ed8f2230e3cb/field/src/helpers.rs#L149>
pub fn split_32<C: Config>(builder: &mut Builder<C>, val: Var<C::N>, n: usize) -> Vec<Felt<C::F>> {
    let felts = builder.var_to_64bits_f_circuit(val);
    assert!(n <= felts.len());
    felts[0..n].to_vec()
}

/// Eval two expressions, return in the reversed order if cond == 1. Otherwise, return in the
/// original order. This is a helper function for optimal performance.
pub fn cond_eval<C: Config, V: MemVariable<C, Expression: Clone> + CanSelect<C>>(
    builder: &mut Builder<C>,
    cond: Var<C::N>,
    v1: impl Into<V::Expression>,
    v2: impl Into<V::Expression>,
) -> [V; 2] {
    let a: V;
    let b: V;
    if builder.flags.static_only {
        let v1: V = builder.eval(v1.into());
        let v2: V = builder.eval(v2.into());
        a = V::select(builder, cond, v2.clone(), v1.clone());
        b = V::select(builder, cond, v1, v2);
    } else {
        let v1 = v1.into();
        let v2 = v2.into();
        a = builder.uninit();
        b = builder.uninit();
        builder.if_eq(cond, C::N::ONE).then_or_else(
            |builder| {
                builder.assign(&a, v2.clone());
                builder.assign(&b, v1.clone());
            },
            |builder| {
                builder.assign(&a, v1.clone());
                builder.assign(&b, v2.clone());
            },
        );
    }
    [a, b]
}
