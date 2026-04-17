use std::{collections::HashSet, fs, io::Write, path::Path, sync::Arc};

use eyre::{Result, WrapErr};
use openvm_recursion_circuit::{
    batch_constraint::{
        eq_airs::{
            eq_3b::Eq3bColumns,
            eq_neg::EqNegCols,
            eq_ns::EqNsColumns,
            eq_sharp_uni::{EqSharpUniCols, EqSharpUniReceiverCols},
            eq_uni::EqUniCols,
        },
        expr_eval::{
            constraints_folding::ConstraintsFoldingCols,
            interactions_folding::InteractionsFoldingCols,
            symbolic_expression::{
                CachedSymbolicExpressionColumns, SingleMainSymbolicExpressionColumns,
            },
        },
        expression_claim::ExpressionClaimCols,
        fractions_folder::FractionsFolderCols,
        sumcheck::{multilinear::MultilinearSumcheckCols, univariate::UnivariateSumcheckCols},
    },
    gkr::{
        input::GkrInputCols, layer::GkrLayerCols, sumcheck::GkrLayerSumcheckCols,
        xi_sampler::GkrXiSamplerCols,
    },
    primitives::{exp_bits_len::ExpBitsLenCols, pow::PowerCheckerCols, range::RangeCheckerCols},
    proof_shape::{proof_shape::ProofShapeCols, pvs::PublicValuesCols},
    stacking::{
        claims::StackingClaimsCols, eq_base::EqBaseCols, eq_bits::EqBitsCols,
        opening::OpeningClaimsCols, sumcheck::SumcheckRoundsCols, univariate::UnivariateRoundCols,
    },
    system::{AggregationSubCircuit, VerifierConfig, VerifierSubCircuit},
    transcript::{
        merkle_verify::MerkleVerifyCols, poseidon2::Poseidon2Cols, transcript::TranscriptCols,
    },
    whir::{
        final_poly_mle_eval::FinalyPolyMleEvalCols, final_poly_query_eval::FinalPolyQueryEvalCols,
        folding::WhirFoldingCols, initial_opened_values::InitialOpenedValuesCols,
        non_initial_opened_values::NonInitialOpenedValuesCols, query::WhirQueryCols,
        sumcheck::SumcheckCols, whir_round::WhirRoundCols,
    },
};
use openvm_stark_backend::{
    lean::{
        extract_constraints_dag_to_lean_writer, generate_lean_air_definition, LeanColumns,
        LeanEntry,
    },
    StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSponge;

use crate::{config, Sdk, SC};

const MAX_NUM_PROOFS: usize = 4;

/// Generates Lean column and constraint files for the verifier recursion circuit AIRs.
pub fn generate_lean_files_for_recursion_circuit<P: AsRef<Path>>(output_dir: P) -> Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).wrap_err("create Lean output dir")?;

    let t0 = std::time::Instant::now();

    eprintln!("[lean_extract] Creating Sdk::standard...");
    let agg_params = config::AggregationSystemParams::default();
    let app_params = openvm_stark_sdk::config::app_params_with_100_bits_security(
        openvm_stark_sdk::config::MAX_APP_LOG_STACKED_HEIGHT,
    );
    let leaf_params = agg_params.leaf.clone();
    let sdk = Sdk::standard(app_params, agg_params);
    eprintln!("[lean_extract] Sdk created in {:?}", t0.elapsed());

    eprintln!("[lean_extract] Generating app_pk / app_vk...");
    let t1 = std::time::Instant::now();
    let app_vk = sdk.app_pk().app_vm_pk.vm_pk.get_vk();
    eprintln!(
        "[lean_extract] app_vk ready in {:?} ({} AIRs)",
        t1.elapsed(),
        app_vk.inner.per_air.len()
    );

    eprintln!("[lean_extract] Building VerifierSubCircuit<{MAX_NUM_PROOFS}>...");
    let t2 = std::time::Instant::now();
    let verifier_config = VerifierConfig {
        continuations_enabled: true,
        final_state_bus_enabled: false,
        has_cached: true,
    };
    let circuit =
        VerifierSubCircuit::<MAX_NUM_PROOFS>::new_with_options(Arc::new(app_vk), verifier_config);
    eprintln!(
        "[lean_extract] VerifierSubCircuit built in {:?}",
        t2.elapsed()
    );

    eprintln!("[lean_extract] Getting AIRs and running keygen...");
    let t3 = std::time::Instant::now();
    let airs = circuit.airs::<SC>();
    eprintln!(
        "[lean_extract] Got {} AIRs in {:?}",
        airs.len(),
        t3.elapsed()
    );

    let t4 = std::time::Instant::now();
    let engine = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine::<
        DuplexSponge,
    >::new(leaf_params);
    let (_pk, vk) = engine.keygen(&airs);
    eprintln!("[lean_extract] Keygen done in {:?}", t4.elapsed());

    eprintln!("[lean_extract] Writing lean files...");
    for (i, (air, air_vk)) in airs.iter().zip(vk.inner.per_air.iter()).enumerate() {
        let air_name = air.name();
        let safe_name = air_name.replace(['<', '>', ',', ' '], "_");

        let t_air = std::time::Instant::now();

        let air_def = lean_air_def_for(&air_name)
            .unwrap_or_else(|| format!("-- Unknown AIR: {air_name} (no LeanColumns mapping)"));
        eprintln!(
            "[lean_extract]   [{i:02}] {air_name} — columns done {:?}",
            t_air.elapsed()
        );

        let header = format!(
            "-- AIR {i}: {air_name}\n-- main_width: {}\n\n",
            air_vk.params.width.main_width()
        );

        let def_path = output_dir.join(format!("{i:02}_{safe_name}_columns.lean"));
        fs::write(&def_path, format!("{header}{air_def}\n"))
            .wrap_err_with(|| format!("write columns file {}", def_path.display()))?;
        eprintln!("[lean_extract]   [{i:02}] columns file written");

        eprintln!(
            "[lean_extract]   [{i:02}] writing DAG ({} nodes, {} constraint roots, {} interactions) to Lean...",
            air_vk.symbolic_constraints.constraints.nodes.len(),
            air_vk.symbolic_constraints.constraints.constraint_idx.len(),
            air_vk.symbolic_constraints.interactions.len(),
        );
        let cst_path = output_dir.join(format!("{i:02}_{safe_name}_constraints.lean"));
        let t_write = std::time::Instant::now();
        let mut cst_file = std::io::BufWriter::new(
            std::fs::File::create(&cst_path)
                .wrap_err_with(|| format!("create constraints file {}", cst_path.display()))?,
        );
        cst_file
            .write_all(header.as_bytes())
            .wrap_err("write constraints header")?;
        extract_constraints_dag_to_lean_writer(
            &air_vk.symbolic_constraints,
            &air_name,
            &mut cst_file,
        )
        .wrap_err("write constraints file")?;
        cst_file.flush().wrap_err("flush constraints file")?;
        let written_bytes = fs::metadata(&cst_path)
            .wrap_err("stat constraints file")?
            .len();
        eprintln!(
            "[lean_extract]   [{i:02}] constraints written in {:?} ({} bytes)",
            t_write.elapsed(),
            written_bytes,
        );

        eprintln!(
            "[lean_extract]   [{i:02}] {air_name} — total {:?}",
            t_air.elapsed(),
        );
    }

    eprintln!(
        "[lean_extract] Done. Wrote {} lean files to {} (total {:?})",
        airs.len(),
        output_dir.display(),
        t0.elapsed(),
    );

    Ok(())
}

fn format_lean_air_name(air_name: &str) -> String {
    let mut formatted = String::with_capacity(air_name.len());
    let mut prev_was_underscore = false;

    for ch in air_name.chars() {
        let replacement = match ch {
            '<' => Some('_'),
            '>' => None,
            ',' | ' ' => Some('_'),
            _ => Some(ch),
        };

        if let Some(ch) = replacement {
            if ch == '_' {
                if prev_was_underscore {
                    continue;
                }
                prev_was_underscore = true;
            } else {
                prev_was_underscore = false;
            }
            formatted.push(ch);
        }
    }

    formatted.trim_end_matches('_').to_string()
}

fn lean_air_def_for(air_name: &str) -> Option<String> {
    let lean_air_name = format_lean_air_name(air_name);
    Some(match air_name {
        n if n.starts_with("SymbolicExpressionAir") => {
            let cached = generate_lean_air_definition_with_subairs::<
                CachedSymbolicExpressionColumns<u8>,
            >(&format!("{lean_air_name}_Cached"));
            let single = generate_lean_air_definition_with_subairs::<
                SingleMainSymbolicExpressionColumns<u8>,
            >(&format!("{lean_air_name}_SingleMain"));
            format!(
                "-- Atypical AIR: width = Cached + SingleMain * {MAX_NUM_PROOFS}\n\
                 -- Cached partition (has_cached=true):\n{cached}\n\n\
                 -- Single main partition (repeated per proof):\n{single}"
            )
        }
        n if n.starts_with("FractionsFolderAir") => {
            generate_lean_air_definition_with_subairs::<FractionsFolderCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("UnivariateSumcheckAir") => {
            generate_lean_air_definition_with_subairs::<UnivariateSumcheckCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("MultilinearSumcheckAir") => {
            generate_lean_air_definition_with_subairs::<MultilinearSumcheckCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqNsAir") => {
            generate_lean_air_definition_with_subairs::<EqNsColumns<u8>>(&lean_air_name)
        }
        n if n.starts_with("Eq3bAir") => {
            generate_lean_air_definition_with_subairs::<Eq3bColumns<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqSharpUniReceiverAir") => {
            generate_lean_air_definition_with_subairs::<EqSharpUniReceiverCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqSharpUniAir") => {
            generate_lean_air_definition_with_subairs::<EqSharpUniCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqUniAir") => {
            generate_lean_air_definition_with_subairs::<EqUniCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("ExpressionClaimAir") => {
            generate_lean_air_definition_with_subairs::<ExpressionClaimCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("InteractionsFoldingAir") => {
            generate_lean_air_definition_with_subairs::<InteractionsFoldingCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("ConstraintsFoldingAir") => {
            generate_lean_air_definition_with_subairs::<ConstraintsFoldingCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqNegAir") => {
            generate_lean_air_definition_with_subairs::<EqNegCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("TranscriptAir") => {
            generate_lean_air_definition_with_subairs::<TranscriptCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("Poseidon2Air") => {
            generate_lean_air_definition_with_subairs::<Poseidon2Cols<u8, 1>>(&lean_air_name)
        }
        n if n.starts_with("MerkleVerifyAir") => {
            generate_lean_air_definition_with_subairs::<MerkleVerifyCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("ProofShapeAir") => {
            let fixed = generate_lean_air_definition_with_subairs::<ProofShapeCols<u8, 4>>(
                &format!("{lean_air_name}_Fixed"),
            );
            format!(
                "-- Atypical AIR: width = ProofShapeCols + idx_encoder.width() + max_cached * DIGEST_SIZE\n\
                 -- Fixed portion (ProofShapeCols<NUM_LIMBS=4>):\n{fixed}"
            )
        }
        n if n.starts_with("PublicValuesAir") => {
            generate_lean_air_definition_with_subairs::<PublicValuesCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("RangeCheckerAir") => {
            generate_lean_air_definition_with_subairs::<RangeCheckerCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("GkrInputAir") => {
            generate_lean_air_definition_with_subairs::<GkrInputCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("GkrLayerSumcheckAir") => {
            generate_lean_air_definition_with_subairs::<GkrLayerSumcheckCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("GkrLayerAir") => {
            generate_lean_air_definition_with_subairs::<GkrLayerCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("GkrXiSamplerAir") => {
            generate_lean_air_definition_with_subairs::<GkrXiSamplerCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("OpeningClaimsAir") => {
            generate_lean_air_definition_with_subairs::<OpeningClaimsCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("UnivariateRoundAir") => {
            generate_lean_air_definition_with_subairs::<UnivariateRoundCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("SumcheckRoundsAir") => {
            generate_lean_air_definition_with_subairs::<SumcheckRoundsCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("StackingClaimsAir") => {
            generate_lean_air_definition_with_subairs::<StackingClaimsCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqBaseAir") => {
            generate_lean_air_definition_with_subairs::<EqBaseCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("EqBitsAir") => {
            generate_lean_air_definition_with_subairs::<EqBitsCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("WhirRoundAir") => {
            let v1 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 1>>(&format!(
                "{lean_air_name}_enc1"
            ));
            let v2 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 2>>(&format!(
                "{lean_air_name}_enc2"
            ));
            let v3 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 3>>(&format!(
                "{lean_air_name}_enc3"
            ));
            format!(
                "-- WhirRoundAir: ENC_WIDTH depends on num_whir_rounds encoder\n{v1}\n\n{v2}\n\n{v3}"
            )
        }
        n if n.starts_with("SumcheckAir") => {
            generate_lean_air_definition_with_subairs::<SumcheckCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("WhirQueryAir") => {
            generate_lean_air_definition_with_subairs::<WhirQueryCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("InitialOpenedValuesAir") => {
            generate_lean_air_definition_with_subairs::<InitialOpenedValuesCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("NonInitialOpenedValuesAir") => {
            generate_lean_air_definition_with_subairs::<NonInitialOpenedValuesCols<u8>>(
                &lean_air_name,
            )
        }
        n if n.starts_with("WhirFoldingAir") => {
            generate_lean_air_definition_with_subairs::<WhirFoldingCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("FinalPolyMleEvalAir") | n.starts_with("Finaly") => {
            generate_lean_air_definition_with_subairs::<FinalyPolyMleEvalCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("FinalPolyQueryEvalAir") => {
            generate_lean_air_definition_with_subairs::<FinalPolyQueryEvalCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("PowerCheckerAir") => {
            generate_lean_air_definition_with_subairs::<PowerCheckerCols<u8>>(&lean_air_name)
        }
        n if n.starts_with("ExpBitsLenAir") => {
            generate_lean_air_definition_with_subairs::<ExpBitsLenCols<u8>>(&lean_air_name)
        }
        _ => return None,
    })
}

fn generate_lean_air_definition_with_subairs<C: LeanColumns>(air_name: &str) -> String {
    let mut seen = HashSet::new();
    let mut defs = Vec::new();
    push_lean_air_definition::<C>(air_name, &mut seen, &mut defs);
    defs.join("\n\n")
}

fn push_lean_air_definition<C: LeanColumns>(
    air_name: &str,
    seen: &mut HashSet<String>,
    defs: &mut Vec<String>,
) {
    let lean_air_name = format_lean_air_name(air_name);
    if !seen.insert(lean_air_name.clone()) {
        return;
    }

    for entry in C::lean_columns() {
        if let LeanEntry::SubAir { type_name, .. } = entry {
            push_known_subair_definition(&type_name, seen, defs);
        }
    }

    defs.push(generate_lean_air_definition::<C>(&lean_air_name));
}

fn push_known_subair_definition(
    type_name: &str,
    seen: &mut HashSet<String>,
    defs: &mut Vec<String>,
) {
    let Some(definition) = define_subair(type_name) else {
        return;
    };

    let lean_air_name = format_lean_air_name(type_name);
    if seen.insert(lean_air_name) {
        defs.push(definition);
    }
}

fn define_subair(type_name: &str) -> Option<String> {
    let lean_air_name = format_lean_air_name(type_name);
    Some(match type_name {
        "IsZeroAuxCols" | "IsEqualAuxCols" => format!(
            "#define_subair \"{lean_air_name}\" using \"openvm_encapsulation\" where\n  Column[\"inv\"]"
        ),
        "Poseidon2SubCols" => {
            let mut lines = vec![format!(
                "#define_subair \"{lean_air_name}\" using \"openvm_encapsulation\" where"
            )];
            lines.push("  Column[\"export_col\"]".to_string());
            for i in 0..16 {
                lines.push(format!("  Column[\"inputs_{i}\"]"));
            }
            for round in 0..4 {
                for lane in 0..16 {
                    lines.push(format!(
                        "  Column[\"beginning_full_rounds_{round}_sbox_{lane}_registers_0\"]"
                    ));
                }
                for lane in 0..16 {
                    lines.push(format!(
                        "  Column[\"beginning_full_rounds_{round}_post_{lane}\"]"
                    ));
                }
            }
            for round in 0..13 {
                lines.push(format!("  Column[\"partial_rounds_{round}_sbox_registers_0\"]"));
                lines.push(format!("  Column[\"partial_rounds_{round}_post\"]"));
            }
            for round in 0..4 {
                for lane in 0..16 {
                    lines.push(format!(
                        "  Column[\"ending_full_rounds_{round}_sbox_{lane}_registers_0\"]"
                    ));
                }
                for lane in 0..16 {
                    lines.push(format!(
                        "  Column[\"ending_full_rounds_{round}_post_{lane}\"]"
                    ));
                }
            }
            lines.join("\n")
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn generates_lean_files_for_recursion_circuit() {
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lean_out");
        generate_lean_files_for_recursion_circuit(output_dir).unwrap();
    }
}
