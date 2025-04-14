use itertools::Itertools;
use openvm_native_compiler::{
    ir::{Builder, Config},
    prelude::*,
};
use openvm_stark_backend::{
    keygen::types::TraceWidth,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_util::log2_strict_usize,
};

use crate::{
    types::{MultiStarkVerificationAdvice, StarkVerificationAdvice},
    vars::{
        LinearConstraintVariable, MultiStarkVerificationAdviceVariable, OptionalVar,
        StarkVerificationAdviceVariable, TraceHeightConstraintSystem, TraceWidthVariable,
    },
};

pub fn get_advice_per_air<C: Config>(
    builder: &mut Builder<C>,
    m_advice: &MultiStarkVerificationAdvice<C>,
    air_ids: &Array<C, Usize<C::N>>,
) -> MultiStarkVerificationAdviceVariable<C> {
    let num_challenges_to_sample_mask = m_advice
        .num_challenges_to_sample
        .iter()
        .map(|&num_challenges_to_sample| vec![builder.eval(RVar::zero()); num_challenges_to_sample])
        .collect_vec();
    let advice_per_air = builder.array(air_ids.len());

    let idx: Usize<_> = builder.eval(RVar::zero());
    for (air_id, advice) in m_advice.per_air.iter().enumerate() {
        builder.if_ne(idx.clone(), air_ids.len()).then(|builder| {
            let curr_air_id = builder.get(air_ids, idx.clone());
            let air_id = Usize::from(air_id);
            builder.if_eq(air_id, curr_air_id).then(|builder| {
                let advice_var = constant_advice_and_update_mask(
                    builder,
                    advice,
                    &num_challenges_to_sample_mask,
                );
                builder.set_value(&advice_per_air, idx.clone(), advice_var);
                builder.inc(&idx);
            });
        });
    }
    // Assert that all AIRs in air_ids are covered.
    // This will ensure that
    // - `air_ids` are in increasing order and that
    // - `air_ids.len() <= m_advice.per_air.len()`.
    builder.assert_var_eq(idx, air_ids.len());

    let trace_height_constraints = m_advice
        .trace_height_constraints
        .iter()
        .map(|constraint| {
            let coefficients = builder.array(constraint.coefficients.len());
            for (i, coeff) in constraint.coefficients.iter().enumerate() {
                let coefficient: Var<_> = builder.constant(C::N::from_canonical_u32(*coeff));
                builder.set(&coefficients, i, coefficient);
            }
            assert!(constraint.threshold <= C::F::ORDER_U32);
            let threshold: Var<_> = builder.constant(C::N::from_wrapped_u32(constraint.threshold));
            let is_threshold_at_p = constraint.threshold == C::F::ORDER_U32;
            LinearConstraintVariable {
                coefficients,
                threshold,
                is_threshold_at_p,
            }
        })
        .collect();

    let height_maxes = builder.array(m_advice.per_air.len());
    for i in 0..m_advice.per_air.len() {
        let max_coefficient = m_advice
            .trace_height_constraints
            .iter()
            .map(|constraint| constraint.coefficients[i])
            .max()
            .unwrap();
        let height_max = if max_coefficient <= 1 {
            OptionalVar {
                is_some: Usize::from(0),
                value: builder.constant(C::N::ZERO),
            }
        } else {
            OptionalVar {
                is_some: Usize::from(1),
                // Because `C::F::ORDER_U32` is prime and `max_coefficient > 1`,
                // `floor(C::F::ORDER_U32 / max_coefficient) * max_coefficient < C::F::ORDER_U32`,
                // `height * max_coefficient` cannot overflow `C::F`.
                value: builder.constant(C::N::from_canonical_u32(
                    C::F::ORDER_U32 / max_coefficient + 1,
                )),
            }
        };
        builder.set(&height_maxes, i, height_max);
    }

    MultiStarkVerificationAdviceVariable {
        per_air: advice_per_air,
        num_challenges_to_sample_mask,
        trace_height_constraint_system: TraceHeightConstraintSystem {
            height_constraints: trace_height_constraints,
            height_maxes,
        },
    }
}

fn constant_advice_and_update_mask<C: Config>(
    builder: &mut Builder<C>,
    advice: &StarkVerificationAdvice<C>,
    num_challenges_to_sample_mask: &[Vec<Usize<C::N>>],
) -> StarkVerificationAdviceVariable<C> {
    let preprocessed_data = if let Some(preprocessed_data) = advice.preprocessed_data.as_ref() {
        let commit = builder.constant(preprocessed_data.commit.clone());
        let arr = builder.array(1);
        builder.set_value(&arr, 0, commit);
        arr
    } else {
        builder.array(0)
    };
    for (phase, &num_challenges) in advice.num_challenges_to_sample.iter().enumerate() {
        for i in 0..num_challenges {
            builder.assign(&num_challenges_to_sample_mask[phase][i], RVar::one());
        }
    }
    StarkVerificationAdviceVariable {
        preprocessed_data,
        width: constant_trace_width(builder, &advice.width),
        log_quotient_degree: builder.eval(RVar::from(log2_strict_usize(advice.quotient_degree))),
        num_public_values: builder.eval(RVar::from(advice.num_public_values)),
        num_challenges_to_sample: constant_usize_array(builder, &advice.num_challenges_to_sample),
        num_exposed_values_after_challenge: constant_usize_array(
            builder,
            &advice.num_exposed_values_after_challenge,
        ),
    }
}

fn constant_trace_width<C: Config>(
    builder: &mut Builder<C>,
    trace_width: &TraceWidth,
) -> TraceWidthVariable<C> {
    TraceWidthVariable {
        preprocessed: constant_usize_option(builder, &trace_width.preprocessed),
        cached_mains: constant_usize_array(builder, &trace_width.cached_mains),
        common_main: builder.eval(RVar::from(trace_width.common_main)),
        after_challenge: constant_usize_array(builder, &trace_width.after_challenge),
    }
}

fn constant_usize_option<C: Config>(
    builder: &mut Builder<C>,
    opt: &Option<usize>,
) -> Array<C, Usize<C::N>> {
    match opt {
        Some(val) => {
            let arr = builder.array(1);
            let v: Usize<_> = builder.eval(RVar::from(*val));
            builder.set_value(&arr, 0, v);
            arr
        }
        None => builder.array(0),
    }
}

fn constant_usize_array<C: Config>(
    builder: &mut Builder<C>,
    arr: &[usize],
) -> Array<C, Usize<C::N>> {
    let carr = builder.array(arr.len());
    for (i, val) in arr.iter().enumerate() {
        let v: Usize<_> = builder.eval(RVar::from(*val));
        builder.set(&carr, i, v);
    }
    carr
}
