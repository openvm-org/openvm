use afs_compiler::{asm::AsmConfig, ir::Builder};
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
