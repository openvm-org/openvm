//! Encoder for the Lean `MultiStarkVerifyingKey` wire type.
//!
//! Field order on the wire follows the Lean struct in
//! `Swirl/Protocol/Noninteractive/VerifyingKey.lean`:
//!
//! - `MultiStarkVerifyingKey`: `inner`, then `preHash`.
//! - `MultiStarkVerifyingKey0`: `params`, `perAir`, `traceHeightConstraints`.
//! - `StarkVerifyingKey`: `preprocessedData`, `params`, `symbolicConstraints`,
//!   `maxConstraintDegree`, `isRequired`, `unusedVariables`.

use std::io::{Result, Write};

use openvm_stark_backend::{
    codec::EncodableConfig,
    interaction::LogUpSecurityParameters,
    keygen::types::{
        LinearConstraint, MultiStarkVerifyingKey, MultiStarkVerifyingKey0, StarkVerifyingKey,
        StarkVerifyingParams, TraceWidth, VerifierSinglePreprocessedData,
    },
    p3_field::PrimeField32,
    SystemParams, WhirConfig, WhirRoundConfig,
};

use super::{
    magic::{write_header, MAGIC_VK},
    primitives::{
        write_bool, write_length_prefix, write_option, write_option_usize, write_u32,
        write_u32_list, write_usize_as_u32,
    },
    symbolic::{write_symbolic_constraints_dag, write_symbolic_variable},
};

/// Encode a verifying key in the field order expected by the Lean decoder.
pub(crate) fn write_vk<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    vk: &MultiStarkVerifyingKey<SC>,
) -> Result<()>
where
    SC::F: PrimeField32,
{
    write_header(writer, MAGIC_VK)?;
    write_vk_inner::<SC, _>(writer, &vk.inner)?;
    SC::encode_digest(&vk.pre_hash, writer)
}

/// Encode the inner system parameters, per-AIR keys, and trace-height constraints.
fn write_vk_inner<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    inner: &MultiStarkVerifyingKey0<SC>,
) -> Result<()>
where
    SC::F: PrimeField32,
{
    write_system_params(writer, &inner.params)?;
    write_length_prefix(writer, inner.per_air.len())?;
    for per_air_vk in &inner.per_air {
        write_stark_verifying_key::<SC, _>(writer, per_air_vk)?;
    }
    write_length_prefix(writer, inner.trace_height_constraints.len())?;
    for lc in &inner.trace_height_constraints {
        write_linear_constraint(writer, lc)?;
    }
    Ok(())
}

/// Encode the proof-system parameters shared by all AIRs.
fn write_system_params<W: Write>(writer: &mut W, params: &SystemParams) -> Result<()> {
    write_usize_as_u32(writer, params.l_skip)?;
    write_usize_as_u32(writer, params.n_stack)?;
    write_usize_as_u32(writer, params.w_stack)?;
    write_usize_as_u32(writer, params.log_blowup)?;
    write_whir_config(writer, &params.whir)?;
    write_logup_security_parameters(writer, &params.logup)?;
    write_usize_as_u32(writer, params.max_constraint_degree)
}

/// Encode the WHIR configuration.
///
/// The Rust `proximity` field is omitted because the Lean wire type has
/// no corresponding field and the verifier does not consume it.
fn write_whir_config<W: Write>(writer: &mut W, whir: &WhirConfig) -> Result<()> {
    write_usize_as_u32(writer, whir.k)?;
    write_length_prefix(writer, whir.rounds.len())?;
    for round in &whir.rounds {
        write_whir_round_config(writer, round)?;
    }
    write_usize_as_u32(writer, whir.mu_pow_bits)?;
    write_usize_as_u32(writer, whir.query_phase_pow_bits)?;
    write_usize_as_u32(writer, whir.folding_pow_bits)
}

/// Encode one WHIR round configuration.
fn write_whir_round_config<W: Write>(writer: &mut W, round: &WhirRoundConfig) -> Result<()> {
    write_usize_as_u32(writer, round.num_queries)
}

/// Encode the LogUp security parameters.
fn write_logup_security_parameters<W: Write>(
    writer: &mut W,
    logup: &LogUpSecurityParameters,
) -> Result<()> {
    write_u32(writer, logup.max_interaction_count)?;
    write_u32(writer, logup.log_max_message_length)?;
    write_usize_as_u32(writer, logup.pow_bits)
}

/// Encode one AIR's verifying key in Lean field order.
fn write_stark_verifying_key<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    vk: &StarkVerifyingKey<SC::F, SC::Digest>,
) -> Result<()>
where
    SC::F: PrimeField32,
{
    write_option(writer, vk.preprocessed_data.as_ref(), |w, pd| {
        write_verifier_single_preprocessed_data::<SC, _>(w, pd)
    })?;
    write_stark_verifying_params(writer, &vk.params)?;
    // The Lean SymbolicConstraintsDag carries `width` and
    // `publicValueCount`; the Rust source struct does not. We supply
    // them from the parent `params` because that is where the Lean
    // `hLayout` / `hPublicValues` invariants source their truth.
    let width = vk.params.width.total_width();
    let public_value_count = vk.params.num_public_values;
    write_symbolic_constraints_dag(writer, width, public_value_count, &vk.symbolic_constraints)?;
    write_usize_as_u32(writer, vk.max_constraint_degree as usize)?;
    write_bool(writer, vk.is_required)?;
    write_length_prefix(writer, vk.unused_variables.len())?;
    for var in &vk.unused_variables {
        write_symbolic_variable(writer, var)?;
    }
    Ok(())
}

/// Encode the optional preprocessed-trace commitment metadata.
fn write_verifier_single_preprocessed_data<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    pd: &VerifierSinglePreprocessedData<SC::Digest>,
) -> Result<()> {
    SC::encode_digest(&pd.commit, writer)?;
    super::primitives::write_isize_as_i32(writer, pd.hypercube_dim)?;
    write_usize_as_u32(writer, pd.stacking_width)
}

/// Encode one AIR's trace layout and public-value parameters.
fn write_stark_verifying_params<W: Write>(
    writer: &mut W,
    params: &StarkVerifyingParams,
) -> Result<()> {
    write_trace_width(writer, &params.width)?;
    write_usize_as_u32(writer, params.num_public_values)?;
    write_bool(writer, params.need_rot)
}

/// Encode a trace layout.
///
/// The Lean `TraceWidth` carries an extra `afterChallenge : List Nat`
/// field with no Rust counterpart, so the encoder writes an empty list.
fn write_trace_width<W: Write>(writer: &mut W, width: &TraceWidth) -> Result<()> {
    write_option_usize(writer, width.preprocessed)?;
    write_length_prefix(writer, width.cached_mains.len())?;
    for w in &width.cached_mains {
        write_usize_as_u32(writer, *w)?;
    }
    write_usize_as_u32(writer, width.common_main)?;
    // afterChallenge
    write_length_prefix(writer, 0)
}

/// Encode one linear trace-height constraint.
fn write_linear_constraint<W: Write>(writer: &mut W, lc: &LinearConstraint) -> Result<()> {
    write_u32_list(writer, &lc.coefficients)?;
    write_u32(writer, lc.threshold)
}
