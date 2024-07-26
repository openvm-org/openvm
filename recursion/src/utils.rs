use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_commit::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, TwoAdicField};

use afs_compiler::asm::AsmConfig;
use afs_compiler::ir::Builder;
use afs_test_utils::config::FriParameters;

use crate::fri::types::FriConfigVariable;
use crate::fri::TwoAdicMultiplicativeCosetVariable;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type RecursionConfig = AsmConfig<Val, Challenge>;
type RecursionBuilder = Builder<RecursionConfig>;

pub fn const_fri_config(
    builder: &mut RecursionBuilder,
    params: &FriParameters,
) -> FriConfigVariable<RecursionConfig> {
    let two_addicity = Val::TWO_ADICITY;
    let mut generators = builder.dyn_array(two_addicity);
    let mut subgroups = builder.dyn_array(two_addicity);
    for i in 0..two_addicity {
        let constant_generator = Val::two_adic_generator(i);
        builder.set(&mut generators, i, constant_generator);

        let constant_domain = TwoAdicMultiplicativeCoset {
            log_n: i,
            shift: Val::one(),
        };
        let domain_value: TwoAdicMultiplicativeCosetVariable<_> = builder.constant(constant_domain);
        builder.set(&mut subgroups, i, domain_value);
    }
    FriConfigVariable {
        log_blowup: builder.eval(BabyBear::from_canonical_usize(params.log_blowup)),
        blowup: builder.eval(BabyBear::from_canonical_usize(1 << params.log_blowup)),
        num_queries: builder.eval(BabyBear::from_canonical_usize(params.num_queries)),
        proof_of_work_bits: builder.eval(BabyBear::from_canonical_usize(params.proof_of_work_bits)),
        subgroups,
        generators,
    }
}

#[allow(dead_code)]
pub fn static_const_fri_config(
    builder: &mut RecursionBuilder,
    params: &FriParameters,
) -> FriConfigVariable<RecursionConfig> {
    let two_addicity = Val::TWO_ADICITY;
    let generators = (0..two_addicity)
        .map(|i| builder.constant(Val::two_adic_generator(i)))
        .collect_vec();
    let subgroups = (0..two_addicity)
        .map(|i| {
            let constant_domain = TwoAdicMultiplicativeCoset {
                log_n: i,
                shift: Val::one(),
            };
            builder.constant(constant_domain)
        })
        .collect_vec();
    FriConfigVariable {
        log_blowup: builder.eval(BabyBear::from_canonical_usize(params.log_blowup)),
        blowup: builder.eval(BabyBear::from_canonical_usize(1 << params.log_blowup)),
        num_queries: builder.eval(BabyBear::from_canonical_usize(params.num_queries)),
        proof_of_work_bits: builder.eval(BabyBear::from_canonical_usize(params.proof_of_work_bits)),
        subgroups: builder.vec(subgroups),
        generators: builder.vec(generators),
    }
}
