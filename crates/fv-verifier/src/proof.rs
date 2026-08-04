//! Encoder for `Proof<SC>` matching
//! `notes/lean-verifier-wire-format.md §B`.
//!
//! Field order on the wire follows the Lean struct in
//! `Swirl/Protocol/Noninteractive/Proof.lean` (6 fields):
//! `commonMainCommit`, `traceVdata`, `gkrProof`,
//! `batchConstraintProof`, `stackingProof`, `whirProof`. The Rust
//! `Proof<SC>::public_values` field is **not** part of this blob — it is
//! emitted via [`super::write_public_values`] as a separate `PUBV`
//! stream.

use std::io::{Result, Write};

use openvm_stark_backend::{
    codec::EncodableConfig,
    proof::{
        BatchConstraintProof, GkrLayerClaims, GkrProof, Proof, StackingProof, TraceVData, WhirProof,
    },
};

use super::{
    magic::{write_header, MAGIC_PROOF},
    primitives::{write_length_prefix, write_option},
};

/// `notes/lean-verifier-wire-format.md §B`.
pub fn write_proof<SC: EncodableConfig, W: Write>(writer: &mut W, proof: &Proof<SC>) -> Result<()> {
    write_header(writer, MAGIC_PROOF)?;
    // §B.1.1
    SC::encode_digest(&proof.common_main_commit, writer)?;
    // §B.1.2 — list of Option<TraceVData>
    write_length_prefix(writer, proof.trace_vdata.len())?;
    for vdata in &proof.trace_vdata {
        write_option(writer, vdata.as_ref(), |w, v| {
            write_trace_vdata::<SC, _>(w, v)
        })?;
    }
    // §B.1.3-6. The Lean `StackingProof` carries `muPowWitness`; on the
    // Rust side that PoW witness lives inside `WhirProof::mu_pow_witness`.
    // The wire follows the Lean layout, so we pass the Rust witness from
    // `proof.whir_proof.mu_pow_witness` into the stacking encoder and
    // omit it from the WHIR body. (See `notes/lean-verifier-wire-format.md
    // §B.5` and §B.6.)
    write_gkr_proof::<SC, _>(writer, &proof.gkr_proof)?;
    write_batch_constraint_proof::<SC, _>(writer, &proof.batch_constraint_proof)?;
    write_stacking_proof::<SC, _>(
        writer,
        &proof.stacking_proof,
        &proof.whir_proof.mu_pow_witness,
    )?;
    write_whir_proof::<SC, _>(writer, &proof.whir_proof)
}

/// `notes/lean-verifier-wire-format.md §B.2`.
fn write_trace_vdata<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    vdata: &TraceVData<SC>,
) -> Result<()> {
    write_length_prefix(writer, vdata.log_height)?;
    write_length_prefix(writer, vdata.cached_commitments.len())?;
    for digest in &vdata.cached_commitments {
        SC::encode_digest(digest, writer)?;
    }
    Ok(())
}

/// `notes/lean-verifier-wire-format.md §B.3.a`.
fn write_gkr_layer_claims<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    claims: &GkrLayerClaims<SC>,
) -> Result<()> {
    SC::encode_extension_field(&claims.p_xi_0, writer)?;
    SC::encode_extension_field(&claims.p_xi_1, writer)?;
    SC::encode_extension_field(&claims.q_xi_0, writer)?;
    SC::encode_extension_field(&claims.q_xi_1, writer)
}

/// `notes/lean-verifier-wire-format.md §B.3`.
fn write_gkr_proof<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    gkr: &GkrProof<SC>,
) -> Result<()> {
    // §B.3.1
    SC::encode_base_field(&gkr.logup_pow_witness, writer)?;
    // §B.3.2
    SC::encode_extension_field(&gkr.q0_claim, writer)?;
    // §B.3.3
    write_length_prefix(writer, gkr.claims_per_layer.len())?;
    for claims in &gkr.claims_per_layer {
        write_gkr_layer_claims::<SC, _>(writer, claims)?;
    }
    // §B.3.4 — Vec<Vec<[EF; 3]>>
    write_length_prefix(writer, gkr.sumcheck_polys.len())?;
    for layer in &gkr.sumcheck_polys {
        write_length_prefix(writer, layer.len())?;
        for evals in layer {
            for value in evals {
                SC::encode_extension_field(value, writer)?;
            }
        }
    }
    Ok(())
}

/// `notes/lean-verifier-wire-format.md §B.4`.
fn write_batch_constraint_proof<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    bcp: &BatchConstraintProof<SC>,
) -> Result<()> {
    write_length_prefix(writer, bcp.numerator_term_per_air.len())?;
    for v in &bcp.numerator_term_per_air {
        SC::encode_extension_field(v, writer)?;
    }
    write_length_prefix(writer, bcp.denominator_term_per_air.len())?;
    for v in &bcp.denominator_term_per_air {
        SC::encode_extension_field(v, writer)?;
    }
    write_length_prefix(writer, bcp.univariate_round_coeffs.len())?;
    for v in &bcp.univariate_round_coeffs {
        SC::encode_extension_field(v, writer)?;
    }
    write_length_prefix(writer, bcp.sumcheck_round_polys.len())?;
    for round in &bcp.sumcheck_round_polys {
        write_length_prefix(writer, round.len())?;
        for v in round {
            SC::encode_extension_field(v, writer)?;
        }
    }
    write_length_prefix(writer, bcp.column_openings.len())?;
    for per_air in &bcp.column_openings {
        write_length_prefix(writer, per_air.len())?;
        for part in per_air {
            write_length_prefix(writer, part.len())?;
            for v in part {
                SC::encode_extension_field(v, writer)?;
            }
        }
    }
    Ok(())
}

/// `notes/lean-verifier-wire-format.md §B.5`. `mu_pow_witness` comes
/// from the Rust `WhirProof` because that is where upstream stores it
/// (the Lean `StackingProof` carries it instead — see the comment on
/// the call site in [`write_proof`]).
fn write_stacking_proof<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    sp: &StackingProof<SC>,
    mu_pow_witness: &SC::F,
) -> Result<()> {
    SC::encode_base_field(mu_pow_witness, writer)?;
    write_length_prefix(writer, sp.univariate_round_coeffs.len())?;
    for v in &sp.univariate_round_coeffs {
        SC::encode_extension_field(v, writer)?;
    }
    write_length_prefix(writer, sp.sumcheck_round_polys.len())?;
    for arr in &sp.sumcheck_round_polys {
        for v in arr {
            SC::encode_extension_field(v, writer)?;
        }
    }
    write_length_prefix(writer, sp.stacking_openings.len())?;
    for opening in &sp.stacking_openings {
        write_length_prefix(writer, opening.len())?;
        for v in opening {
            SC::encode_extension_field(v, writer)?;
        }
    }
    Ok(())
}

/// `notes/lean-verifier-wire-format.md §B.6`.
fn write_whir_proof<SC: EncodableConfig, W: Write>(
    writer: &mut W,
    whir: &WhirProof<SC>,
) -> Result<()> {
    // The Lean field order is:
    //   whirSumcheckPolys, codewordCommits, oodValues,
    //   foldingPowWitnesses, queryPhasePowWitnesses,
    //   initialRoundOpenedRows, initialRoundMerkleProofs,
    //   codewordOpenedValues, codewordMerkleProofs, finalPoly
    // `mu_pow_witness` is intentionally NOT in this body: the Lean
    // type tree carries it on `StackingProof` (see the call in
    // `write_proof`), so the WHIR body skips it.
    write_length_prefix(writer, whir.whir_sumcheck_polys.len())?;
    for arr in &whir.whir_sumcheck_polys {
        for v in arr {
            SC::encode_extension_field(v, writer)?;
        }
    }
    write_length_prefix(writer, whir.codeword_commits.len())?;
    for d in &whir.codeword_commits {
        SC::encode_digest(d, writer)?;
    }
    write_length_prefix(writer, whir.ood_values.len())?;
    for v in &whir.ood_values {
        SC::encode_extension_field(v, writer)?;
    }
    write_length_prefix(writer, whir.folding_pow_witnesses.len())?;
    for v in &whir.folding_pow_witnesses {
        SC::encode_base_field(v, writer)?;
    }
    write_length_prefix(writer, whir.query_phase_pow_witnesses.len())?;
    for v in &whir.query_phase_pow_witnesses {
        SC::encode_base_field(v, writer)?;
    }
    // initialRoundOpenedRows: List (List (List (List F)))
    write_length_prefix(writer, whir.initial_round_opened_rows.len())?;
    for per_commit in &whir.initial_round_opened_rows {
        write_length_prefix(writer, per_commit.len())?;
        for per_query in per_commit {
            write_length_prefix(writer, per_query.len())?;
            for per_row in per_query {
                write_length_prefix(writer, per_row.len())?;
                for v in per_row {
                    SC::encode_base_field(v, writer)?;
                }
            }
        }
    }
    // initialRoundMerkleProofs: List (List (List Digest))  -- MerkleProof = List Digest
    write_length_prefix(writer, whir.initial_round_merkle_proofs.len())?;
    for per_commit in &whir.initial_round_merkle_proofs {
        write_length_prefix(writer, per_commit.len())?;
        for proof in per_commit {
            write_length_prefix(writer, proof.len())?;
            for d in proof {
                SC::encode_digest(d, writer)?;
            }
        }
    }
    // codewordOpenedValues: List (List (List EF))
    write_length_prefix(writer, whir.codeword_opened_values.len())?;
    for round in &whir.codeword_opened_values {
        write_length_prefix(writer, round.len())?;
        for q in round {
            write_length_prefix(writer, q.len())?;
            for v in q {
                SC::encode_extension_field(v, writer)?;
            }
        }
    }
    // codewordMerkleProofs: List (List (List Digest))
    write_length_prefix(writer, whir.codeword_merkle_proofs.len())?;
    for round in &whir.codeword_merkle_proofs {
        write_length_prefix(writer, round.len())?;
        for proof in round {
            write_length_prefix(writer, proof.len())?;
            for d in proof {
                SC::encode_digest(d, writer)?;
            }
        }
    }
    // finalPoly: List EF
    write_length_prefix(writer, whir.final_poly.len())?;
    for v in &whir.final_poly {
        SC::encode_extension_field(v, writer)?;
    }
    Ok(())
}
