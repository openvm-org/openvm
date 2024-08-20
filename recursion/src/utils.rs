use afs_compiler::{
    asm::AsmConfig,
    ir::{Builder, Config, Felt, Var},
};
use afs_test_utils::config::FriParameters;
use p3_baby_bear::BabyBear;
use p3_commit::TwoAdicMultiplicativeCoset;
use p3_field::{extension::BinomialExtensionField, AbstractField, TwoAdicField};

use crate::fri::{types::FriConfigVariable, TwoAdicMultiplicativeCosetVariable};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type RecursionConfig = AsmConfig<Val, Challenge>;
type RecursionBuilder = Builder<RecursionConfig>;

pub fn const_fri_config(
    builder: &mut RecursionBuilder,
    params: &FriParameters,
) -> FriConfigVariable<RecursionConfig> {
    let two_adicity = Val::TWO_ADICITY;
    let mut generators = builder.array(two_adicity);
    let mut subgroups = builder.array(two_adicity);
    for i in 0..Val::TWO_ADICITY {
        let constant_generator = Val::two_adic_generator(i);
        builder.set(&mut generators, i, constant_generator);

        let constant_domain = TwoAdicMultiplicativeCoset {
            log_n: i,
            shift: Val::one(),
        };
        let domain_value: TwoAdicMultiplicativeCosetVariable<_> = builder.constant(constant_domain);
        // FIXME: here must use `builder.set_value`. `builder.set` will convert `Usize::Const`
        // to `Usize::Var` because it calls `builder.eval`.
        builder.set_value(&mut subgroups, i, domain_value);
    }
    FriConfigVariable {
        log_blowup: params.log_blowup,
        blowup: 1 << params.log_blowup,
        num_queries: params.num_queries,
        proof_of_work_bits: params.proof_of_work_bits,
        subgroups,
        generators,
    }
}

/// Reference: https://github.com/Plonky3/Plonky3/blob/622375885320ac6bf3c338001760ed8f2230e3cb/field/src/helpers.rs#L136
pub fn reduce_32<C: Config>(builder: &mut Builder<C>, vals: &[Felt<C::F>]) -> Var<C::N> {
    let mut power = C::N::one();
    let result: Var<C::N> = builder.eval(C::N::zero());
    for val in vals.iter() {
        let bits = builder.num2bits_f_circuit(*val);
        let val = builder.bits2num_v_circuit(&bits);
        builder.assign(&result, result + val * power);
        power *= C::N::from_canonical_usize(1usize << 32);
    }
    result
}

/// Reference: https://github.com/Plonky3/Plonky3/blob/622375885320ac6bf3c338001760ed8f2230e3cb/field/src/helpers.rs#L149
pub fn split_32<C: Config>(builder: &mut Builder<C>, val: Var<C::N>, n: usize) -> Vec<Felt<C::F>> {
    let bits = builder.num2bits_v_circuit(val, 256);
    let mut results = Vec::new();
    for i in 0..n {
        let result: Felt<C::F> = builder.eval(C::F::zero());
        for j in 0..64 {
            let bit = bits[i * 64 + j];
            let t = builder.eval(result + C::F::from_wrapped_u64(1 << j));
            let z = builder.select_f(bit, t, result);
            builder.assign(&result, z);
        }
        results.push(result);
    }
    results
}
