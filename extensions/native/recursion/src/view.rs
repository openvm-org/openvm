use ax_stark_backend::{keygen::types::TraceWidth, p3_util::log2_strict_usize};
use axvm_native_compiler::{
    ir::{Builder, Config},
    prelude::*,
};
use itertools::Itertools;

use crate::{
    types::{MultiStarkVerificationAdvice, StarkVerificationAdvice},
    vars::{
        MultiStarkVerificationAdviceVariable, StarkVerificationAdviceVariable, TraceWidthVariable,
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
        let curr_air_id = builder.get(air_ids, idx.clone());
        let air_id = Usize::from(air_id);
        builder.if_eq(air_id, curr_air_id).then(|builder| {
            let advice_var =
                constant_advice_and_update_mask(builder, advice, &num_challenges_to_sample_mask);
            builder.set_value(&advice_per_air, idx.clone(), advice_var);
            builder.inc(&idx);
        });
    }

    MultiStarkVerificationAdviceVariable {
        per_air: advice_per_air,
        num_challenges_to_sample_mask,
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
