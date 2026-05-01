//! Driver: walk the recursion verifier circuit, render every supported
//! AIR's symbolic constraints, and emit per-AIR Lean modules in the
//! `Fundamentals.Air` dialect under
//! `<output>/Recursion/<Group>/<AirStem>/Generated/{Schema,Constraints,Interactions}.lean`.
//!
//! V1 scope: cached partitions, public values, preprocessed columns,
//! and atypical multi-partition AIRs (`SymbolicExpressionAir`,
//! `ProofShapeAir`, `WhirRoundAir`, `Poseidon2Air`) are skipped with a
//! warning. The leaf of `vk.symbolic_constraints` for those AIRs still
//! mentions buses we don't have a name for; unknown bus indices fall
//! back to `bus_<idx>` placeholder constructors.

use std::{
    collections::BTreeSet,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use eyre::{bail, Result, WrapErr};
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
    transcript::{merkle_verify::MerkleVerifyCols, transcript::TranscriptCols},
    whir::{
        final_poly_mle_eval::FinalyPolyMleEvalCols, final_poly_query_eval::FinalPolyQueryEvalCols,
        folding::WhirFoldingCols, initial_opened_values::InitialOpenedValuesCols,
        non_initial_opened_values::NonInitialOpenedValuesCols, query::WhirQueryCols,
        sumcheck::SumcheckCols, whir_round::WhirRoundCols,
    },
};
use openvm_stark_backend::{
    interaction::BusIndex,
    lean::{
        air_file_stem, flat_columns_of, render_air, write_constraints, write_interactions,
        write_schema, BusBinding, LeanRenderOptions, LeanWriteOptions, RenderedAir,
    },
    StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSponge;

use crate::{config, Sdk, SC};

const MAX_NUM_PROOFS: usize = 4;
const ROOT_MODULE: &str = "Recursion";
const FUNDAMENTALS_IMPORT: &str = "Fundamentals.Air";
const BUS_DEFS_IMPORT: &str = "Recursion.BusDefs";
const BUS_DEFS_NAMESPACE: &str = "Recursion.BusDefs";
const BABY_BEAR_P: u32 = 2_013_265_921;

/// Map a VK bus index to its `BusIdx` constructor name. Order tracks
/// `recursion::system::BusInventory::new` (external buses), then the
/// per-module internal buses in the order they're constructed in the
/// `*Module::new` functions. Unknown indices fall back to
/// `bus_<idx>` at the writer level.
fn standard_bus_table() -> Vec<BusBinding> {
    let names: &[&str] = &[
        // External (system::BusInventory::new)
        "transcriptBus",                  // 0
        "poseidon2PermuteBus",            // 1
        "poseidon2CompressBus",           // 2
        "merkleVerifyBus",                // 3
        "gkrModuleBus",                   // 4
        "batchConstraintModuleBus",       // 5
        "stackingModuleBus",              // 6
        "whirModuleBus",                  // 7
        "whirMuBus",                      // 8
        "airShapeBus",                    // 9
        "airPresenceBus",                 // 10
        "hyperdimBus",                    // 11
        "liftedHeightsBus",               // 12
        "stackingIndicesBus",             // 13
        "commitmentsBus",                 // 14
        "publicValuesBus",                // 15
        "selUniBus",                      // 16
        "rangeCheckerBus",                // 17
        "powerCheckerBus",                // 18
        "expressionClaimNMaxBus",         // 19
        "constraintsFoldingInputBus",     // 20
        "interactionsFoldingInputBus",    // 21
        "fractionFolderInputBus",         // 22
        "nLiftBus",                       // 23
        "eqNLogupNMaxBus",                // 24
        "eq3bShapeBus",                   // 25
        "xiRandomnessBus",                // 26
        "constraintRandomnessBus",        // 27
        "whirOpeningPointBus",            // 28
        "whirOpeningPointLookupBus",      // 29
        "columnClaimsBus",                // 30
        "expBitsLenBus",                  // 31
        "rightShiftBus",                  // 32
        "eqNegBaseRandBus",               // 33
        "eqNegResultBus",                 // 34
        "cachedCommitBus",                // 35
        "preHashBus",                     // 36
        "finalStateBus",                  // 37
    ];
    // Per-module internal buses, in module instantiation order:
    // transcript (0 buses), proof_shape (3), gkr (6), batch_constraint
    // (13), stacking (8), whir (12). Total: 38 + 0 + 3 + 6 + 13 + 8 + 12 = 80.
    let module_internal_offset = 38;
    let module_internal_names: &[&str] = &[
        // ProofShapeModule (38–40)
        "proofShapePermutationBus",   // 38
        "startingTidxBus",            // 39
        "numPvsBus",                  // 40
        // GkrModule (41–46)
        "gkrLayerInputBus",           // 41
        "gkrLayerOutputBus",          // 42
        "gkrSumcheckInputBus",        // 43
        "gkrSumcheckOutputBus",       // 44
        "gkrSumcheckChallengeBus",    // 45
        "gkrXiSamplerBus",            // 46
        // BatchConstraintModule (47–59)
        "batchConstraintConductorBus", // 47
        "univariateSumcheckInputBus",  // 48
        "sumcheckClaimBus",            // 49
        "eqZeroNBus",                  // 50
        "eqSharpUniBus",               // 51
        "eq3bBus",                     // 52
        "eqNegInternalBus",            // 53
        "selHypercubeBus",             // 54
        "eqNOuterBus",                 // 55
        "symbolicExpressionBus",       // 56
        "expressionClaimBus",          // 57
        "interactionsFoldingBus",      // 58
        "constraintsFoldingBus",       // 59
        // StackingModule (60–67)
        "stackingTidxBus",             // 60
        "claimCoefficientsBus",        // 61
        "sumcheckClaimsBus",           // 62
        "eqRandValuesBus",             // 63
        "eqBaseBus",                   // 64
        "eqBitsInternalBus",           // 65
        "eqKernelLookupBus",           // 66
        "eqBitsLookupBus",             // 67
        // WhirModule (68–79)
        "whirSumcheckBus",             // 68
        "whirAlphaBus",                // 69
        "whirGammaBus",                // 70
        "whirQueryBus",                // 71
        "verifyQueriesBus",            // 72
        "verifyQueryBus",              // 73
        "whirEqAlphaUBus",             // 74
        "whirFoldingBus",              // 75
        "finalPolyMleEvalBus",         // 76
        "finalPolyQueryEvalBus",       // 77
        "whirFinalPolyBus",            // 78
        "finalPolyFoldingBus",         // 79
    ];
    let mut bindings: Vec<BusBinding> = names
        .iter()
        .enumerate()
        .map(|(i, n)| BusBinding {
            vk_index: i as BusIndex,
            lean_name: (*n).to_string(),
        })
        .collect();
    bindings.extend(
        module_internal_names
            .iter()
            .enumerate()
            .map(|(i, n)| BusBinding {
                vk_index: (module_internal_offset + i) as BusIndex,
                lean_name: (*n).to_string(),
            }),
    );
    bindings
}

/// Group classification: which subdirectory of `Recursion/` the AIR
/// goes into. Determines its Lean namespace.
fn air_group(air_name: &str) -> Result<&'static str> {
    Ok(match air_name {
        n if n.starts_with("ConstraintsFoldingAir")
            || n.starts_with("Eq3bAir")
            || n.starts_with("EqNegAir")
            || n.starts_with("EqNsAir")
            || n.starts_with("EqSharpUniAir")
            || n.starts_with("EqSharpUniReceiverAir")
            || n.starts_with("EqUniAir")
            || n.starts_with("ExpressionClaimAir")
            || n.starts_with("FractionsFolderAir")
            || n.starts_with("InteractionsFoldingAir")
            || n.starts_with("MultilinearSumcheckAir")
            || n.starts_with("SymbolicExpressionAir")
            || n.starts_with("UnivariateSumcheckAir") =>
        {
            "BatchConstraint"
        }
        n if n.starts_with("GkrInputAir")
            || n.starts_with("GkrLayerAir")
            || n.starts_with("GkrLayerSumcheckAir")
            || n.starts_with("GkrXiSamplerAir") =>
        {
            "GKR"
        }
        n if n.starts_with("ExpBitsLenAir")
            || n.starts_with("PowerCheckerAir")
            || n.starts_with("RangeCheckerAir") =>
        {
            "Primitive"
        }
        n if n.starts_with("ProofShapeAir") || n.starts_with("PublicValuesAir") => "ProofShape",
        n if n.starts_with("EqBaseAir")
            || n.starts_with("EqBitsAir")
            || n.starts_with("OpeningClaimsAir")
            || n.starts_with("StackingClaimsAir")
            || n.starts_with("SumcheckRoundsAir")
            || n.starts_with("UnivariateRoundAir") =>
        {
            "Stacking"
        }
        n if n.starts_with("MerkleVerifyAir")
            || n.starts_with("Poseidon2Air")
            || n.starts_with("TranscriptAir") =>
        {
            "Transcript"
        }
        n if n.starts_with("FinalPolyMleEvalAir")
            || n.starts_with("Finaly")
            || n.starts_with("FinalPolyQueryEvalAir")
            || n.starts_with("InitialOpenedValuesAir")
            || n.starts_with("NonInitialOpenedValuesAir")
            || n.starts_with("SumcheckAir")
            || n.starts_with("WhirFoldingAir")
            || n.starts_with("WhirQueryAir")
            || n.starts_with("WhirRoundAir") =>
        {
            "WHIR"
        }
        _ => bail!("unknown recursion AIR group for {air_name}"),
    })
}

/// Resolve an AIR's Cols struct to a flat column-name list. Returns
/// `None` for AIRs whose layout this v1 driver doesn't yet support
/// (multi-partition, encoder-parameterized, etc.).
fn flat_column_names(air_name: &str) -> Option<Vec<String>> {
    Some(match air_name {
        n if n.starts_with("FractionsFolderAir") => flat_columns_of::<FractionsFolderCols<u8>>(),
        n if n.starts_with("UnivariateSumcheckAir") => {
            flat_columns_of::<UnivariateSumcheckCols<u8>>()
        }
        n if n.starts_with("MultilinearSumcheckAir") => {
            flat_columns_of::<MultilinearSumcheckCols<u8>>()
        }
        n if n.starts_with("EqNsAir") => flat_columns_of::<EqNsColumns<u8>>(),
        n if n.starts_with("Eq3bAir") => flat_columns_of::<Eq3bColumns<u8>>(),
        n if n.starts_with("EqSharpUniReceiverAir") => {
            flat_columns_of::<EqSharpUniReceiverCols<u8>>()
        }
        n if n.starts_with("EqSharpUniAir") => flat_columns_of::<EqSharpUniCols<u8>>(),
        n if n.starts_with("EqUniAir") => flat_columns_of::<EqUniCols<u8>>(),
        n if n.starts_with("ExpressionClaimAir") => flat_columns_of::<ExpressionClaimCols<u8>>(),
        n if n.starts_with("InteractionsFoldingAir") => {
            flat_columns_of::<InteractionsFoldingCols<u8>>()
        }
        n if n.starts_with("ConstraintsFoldingAir") => {
            flat_columns_of::<ConstraintsFoldingCols<u8>>()
        }
        n if n.starts_with("EqNegAir") => flat_columns_of::<EqNegCols<u8>>(),
        n if n.starts_with("TranscriptAir") => flat_columns_of::<TranscriptCols<u8>>(),
        n if n.starts_with("MerkleVerifyAir") => flat_columns_of::<MerkleVerifyCols<u8>>(),
        n if n.starts_with("PublicValuesAir") => flat_columns_of::<PublicValuesCols<u8>>(),
        n if n.starts_with("RangeCheckerAir") => flat_columns_of::<RangeCheckerCols<u8>>(),
        n if n.starts_with("GkrInputAir") => flat_columns_of::<GkrInputCols<u8>>(),
        n if n.starts_with("GkrLayerSumcheckAir") => flat_columns_of::<GkrLayerSumcheckCols<u8>>(),
        n if n.starts_with("GkrLayerAir") => flat_columns_of::<GkrLayerCols<u8>>(),
        n if n.starts_with("GkrXiSamplerAir") => flat_columns_of::<GkrXiSamplerCols<u8>>(),
        n if n.starts_with("OpeningClaimsAir") => flat_columns_of::<OpeningClaimsCols<u8>>(),
        n if n.starts_with("UnivariateRoundAir") => flat_columns_of::<UnivariateRoundCols<u8>>(),
        n if n.starts_with("SumcheckRoundsAir") => flat_columns_of::<SumcheckRoundsCols<u8>>(),
        n if n.starts_with("StackingClaimsAir") => flat_columns_of::<StackingClaimsCols<u8>>(),
        n if n.starts_with("EqBaseAir") => flat_columns_of::<EqBaseCols<u8>>(),
        n if n.starts_with("EqBitsAir") => flat_columns_of::<EqBitsCols<u8>>(),
        n if n.starts_with("SumcheckAir") => flat_columns_of::<SumcheckCols<u8>>(),
        n if n.starts_with("WhirQueryAir") => flat_columns_of::<WhirQueryCols<u8>>(),
        n if n.starts_with("InitialOpenedValuesAir") => {
            flat_columns_of::<InitialOpenedValuesCols<u8>>()
        }
        n if n.starts_with("NonInitialOpenedValuesAir") => {
            flat_columns_of::<NonInitialOpenedValuesCols<u8>>()
        }
        n if n.starts_with("WhirFoldingAir") => flat_columns_of::<WhirFoldingCols<u8>>(),
        n if n.starts_with("FinalPolyMleEvalAir") || n.starts_with("Finaly") => {
            flat_columns_of::<FinalyPolyMleEvalCols<u8>>()
        }
        n if n.starts_with("FinalPolyQueryEvalAir") => {
            flat_columns_of::<FinalPolyQueryEvalCols<u8>>()
        }
        n if n.starts_with("PowerCheckerAir") => flat_columns_of::<PowerCheckerCols<u8>>(),
        n if n.starts_with("ExpBitsLenAir") => flat_columns_of::<ExpBitsLenCols<u8>>(),
        // Hardcoded layouts for AIRs that don't fit the simple
        // `flat_columns_of::<Cols<u8>>` pattern (sub-air-heavy or
        // encoder-parameterized). Names sourced from the previous
        // generator's deprecated output (ws-fv/Recursion/Deprecated/Airs/).
        n if n.starts_with("Poseidon2Air") => poseidon2_air_columns(),
        n if n.starts_with("ProofShapeAir") => proof_shape_air_columns(),
        n if n.starts_with("WhirRoundAir") => whir_round_air_columns(),
        // SymbolicExpressionAir: cached partition + MAX_NUM_PROOFS-fold
        // repeated SingleMain partition. Schema.lean emits a flat
        // singleMain layout (concatenated cached + per-proof
        // SingleMain blocks); the constraint renderer uses
        // partition_offsets to map (part_index, index) to flat name.
        n if n.starts_with("SymbolicExpressionAir") => symbolic_expression_air_columns(),
        _ => return None,
    })
}

fn symbolic_expression_air_columns() -> Vec<String> {
    let cached = flat_columns_of::<CachedSymbolicExpressionColumns<u8>>();
    let single_main = flat_columns_of::<SingleMainSymbolicExpressionColumns<u8>>();
    let mut cols = cached;
    for proof in 0..MAX_NUM_PROOFS {
        for field in &single_main {
            cols.push(format!("{field}_proof_{proof}"));
        }
    }
    cols
}

fn poseidon2_air_columns() -> Vec<String> {
    let mut cols: Vec<String> = Vec::with_capacity(301);
    cols.push("export_col".to_string());
    for i in 0..16 {
        cols.push(format!("inputs_{i}"));
    }
    for round in 0..4 {
        for lane in 0..16 {
            cols.push(format!(
                "beginning_full_rounds_{round}_sbox_{lane}_registers_0"
            ));
        }
        for lane in 0..16 {
            cols.push(format!("beginning_full_rounds_{round}_post_{lane}"));
        }
    }
    for round in 0..13 {
        cols.push(format!("partial_rounds_{round}_sbox_registers_0"));
        cols.push(format!("partial_rounds_{round}_post"));
    }
    for round in 0..4 {
        for lane in 0..16 {
            cols.push(format!(
                "ending_full_rounds_{round}_sbox_{lane}_registers_0"
            ));
        }
        for lane in 0..16 {
            cols.push(format!("ending_full_rounds_{round}_post_{lane}"));
        }
    }
    cols.push("permute_mult".to_string());
    cols.push("compress_mult".to_string());
    debug_assert_eq!(cols.len(), 301);
    cols
}

fn proof_shape_air_columns() -> Vec<String> {
    // 28 fixed columns from ProofShapeCols<u8, 4> + idx_encoder.width()=11
    // (`idx_flags_*`) + max_cached=1 × DIGEST_SIZE=8 (`cached_commits_*_*`) = 47.
    // Encoder width and max_cached are runtime-determined; values here match
    // the standard Sdk config.
    let mut cols = flat_columns_of::<ProofShapeCols<u8, 4>>();
    for i in 0..11 {
        cols.push(format!("idx_flags_{i}"));
    }
    for cached_idx in 0..1 {
        for elem_idx in 0..8 {
            cols.push(format!("cached_commits_{cached_idx}_{elem_idx}"));
        }
    }
    debug_assert_eq!(cols.len(), 47);
    cols
}

fn whir_round_air_columns() -> Vec<String> {
    // 45 base columns + ENC_WIDTH=2 (`whir_round_enc_*`) = 47. The
    // standard Sdk uses the enc2 variant; if a future config picks
    // enc1 or enc3, swap the const generic here.
    flat_columns_of::<WhirRoundCols<u8, 2>>()
}

/// Generates Lean files for the verifier recursion circuit AIRs under a
/// `Recursion` tree at `output_dir`.
pub fn generate_lean_files_for_recursion_circuit<P: AsRef<Path>>(output_dir: P) -> Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).wrap_err("create Lean output dir")?;

    let recursion_dir = output_dir.join(ROOT_MODULE);
    fs::create_dir_all(&recursion_dir).wrap_err("create Recursion output dir")?;

    eprintln!("[lean] Building Sdk::standard...");
    let agg_params = config::AggregationSystemParams::default();
    let app_params = openvm_stark_sdk::config::app_params_with_100_bits_security(
        openvm_stark_sdk::config::MAX_APP_LOG_STACKED_HEIGHT,
    );
    let leaf_params = agg_params.leaf.clone();
    let sdk = Sdk::standard(app_params, agg_params);

    let app_vk = sdk.app_pk().app_vm_pk.vm_pk.get_vk();
    eprintln!(
        "[lean] app_vk has {} AIRs",
        app_vk.inner.per_air.len()
    );

    let verifier_config = VerifierConfig {
        continuations_enabled: true,
        final_state_bus_enabled: false,
        has_cached: true,
    };
    let circuit =
        VerifierSubCircuit::<MAX_NUM_PROOFS>::new_with_options(Arc::new(app_vk), verifier_config);
    let airs = circuit.airs::<SC>();
    eprintln!("[lean] VerifierSubCircuit has {} AIRs", airs.len());

    let engine = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine::<
        DuplexSponge,
    >::new(leaf_params);
    let (_pk, vk) = engine.keygen(&airs);

    let bus_table = standard_bus_table();

    let mut emitted: Vec<EmittedModule> = Vec::new();
    let mut skipped: Vec<(String, &'static str)> = Vec::new();

    for (i, (air, air_vk)) in airs.iter().zip(vk.inner.per_air.iter()).enumerate() {
        let air_name = air.name();
        let group = match air_group(&air_name) {
            Ok(g) => g,
            Err(_) => {
                skipped.push((air_name.clone(), "unknown group"));
                continue;
            }
        };

        // For non-partitioned AIRs (no cached partition), partition_offsets
        // is just `[0]`. For SymbolicExpressionAir we build a partitioned
        // layout (cached at part 0, common-of-MAX_NUM_PROOFS-singles at part 1).
        let cached_widths = &air_vk.params.width.cached_mains;
        let partition_offsets: Vec<usize> = if cached_widths.is_empty() {
            vec![0]
        } else if air_name.starts_with("SymbolicExpressionAir") {
            let mut offsets = Vec::with_capacity(cached_widths.len() + 1);
            let mut acc = 0usize;
            offsets.push(acc);
            for w in cached_widths {
                acc += w;
                offsets.push(acc);
            }
            offsets
        } else {
            skipped.push((air_name.clone(), "cached partition (v1 unsupported)"));
            continue;
        };

        let column_names = match flat_column_names(&air_name) {
            Some(n) => n,
            None => {
                skipped.push((air_name.clone(), "unsupported AIR shape (v1)"));
                continue;
            }
        };
        let expected_width: usize =
            cached_widths.iter().sum::<usize>() + air_vk.params.width.common_main;
        if column_names.len() != expected_width {
            skipped.push((
                air_name.clone(),
                "column count mismatch with main_width",
            ));
            continue;
        }

        let stem = air_file_stem(&air_name).to_string();
        let air_namespace = format!("{ROOT_MODULE}.{group}.{stem}");
        let schema_import = format!("{air_namespace}.Generated.Schema");
        let constraints_import = format!("{air_namespace}.Generated.Constraints");
        let opts = LeanWriteOptions {
            render: LeanRenderOptions::default(),
            characteristic: Some(BABY_BEAR_P),
            fundamentals_import: FUNDAMENTALS_IMPORT,
            bus_defs_import: BUS_DEFS_IMPORT,
            bus_defs_namespace: BUS_DEFS_NAMESPACE,
            air_namespace: air_namespace.clone(),
            schema_import: schema_import.clone(),
            constraints_import: constraints_import.clone(),
            partition_offsets: partition_offsets.clone(),
        };

        let rendered: RenderedAir = match render_air(
            &air_vk.symbolic_constraints,
            &column_names,
            &bus_table,
            &opts,
        ) {
            Ok(r) => r,
            Err(e) => {
                skipped.push((air_name.clone(), Box::leak(format!("{e}").into_boxed_str())));
                continue;
            }
        };

        let stem_dir = recursion_dir.join(group).join(&stem);
        let generated_dir = stem_dir.join("Generated");
        fs::create_dir_all(&generated_dir)
            .wrap_err_with(|| format!("create dir {}", generated_dir.display()))?;

        write_lean_file(generated_dir.join("Schema.lean"), |w| {
            write_schema(w, &air_name, &column_names, &opts)
                .wrap_err("write Schema.lean")?;
            Ok(())
        })?;
        write_lean_file(generated_dir.join("Constraints.lean"), |w| {
            write_constraints(w, &air_name, &column_names, &rendered, &opts)
                .wrap_err("write Constraints.lean")?;
            Ok(())
        })?;
        write_lean_file(generated_dir.join("Interactions.lean"), |w| {
            write_interactions(w, &air_name, &rendered, &opts)
                .wrap_err("write Interactions.lean")?;
            Ok(())
        })?;

        // Hand-written sidecars: only create if not already present so
        // we don't clobber editorial work.
        let labels_path = stem_dir.join("Labels.lean");
        if !labels_path.exists() {
            fs::write(&labels_path, "")
                .wrap_err_with(|| format!("write {}", labels_path.display()))?;
        }
        let facts_path = stem_dir.join("Facts.lean");
        if !facts_path.exists() {
            fs::write(&facts_path, "")
                .wrap_err_with(|| format!("write {}", facts_path.display()))?;
        }

        eprintln!(
            "[lean]   [{i:02}] {air_name} → {}/{}/{{Generated/{{Schema,Constraints,Interactions}},Labels,Facts}}.lean",
            group, stem
        );
        emitted.push(EmittedModule {
            group,
            stem,
            air_namespace,
        });
    }

    write_per_air_module_files(&recursion_dir, &emitted)?;
    write_root_module(output_dir.join(format!("{ROOT_MODULE}.lean")), &emitted)?;

    eprintln!(
        "[lean] Done. Emitted {} AIR(s); skipped {}.",
        emitted.len(),
        skipped.len()
    );
    if !skipped.is_empty() {
        eprintln!("[lean] Skipped:");
        for (name, why) in &skipped {
            eprintln!("[lean]   - {name}: {why}");
        }
    }
    Ok(())
}

struct EmittedModule {
    group: &'static str,
    stem: String,
    air_namespace: String,
}

/// Per-AIR module file `<Group>/<AirStem>.lean`: imports its own five
/// sub-files (Schema, Constraints, Interactions, Labels, Facts).
fn write_per_air_module_files(recursion_dir: &Path, modules: &[EmittedModule]) -> Result<()> {
    for m in modules {
        let path = recursion_dir.join(m.group).join(format!("{}.lean", m.stem));
        let air_ns = &m.air_namespace;
        write_lean_file(path, |w| {
            writeln!(w, "import {air_ns}.Generated.Schema")?;
            writeln!(w, "import {air_ns}.Generated.Constraints")?;
            writeln!(w, "import {air_ns}.Generated.Interactions")?;
            writeln!(w, "import {air_ns}.Labels")?;
            writeln!(w, "import {air_ns}.Facts")?;
            Ok(())
        })?;
    }
    Ok(())
}

/// Root `Recursion.lean`: one import line per AIR module (plus
/// `Recursion.BusDefs`). The per-AIR module files handle their own
/// nested imports.
fn write_root_module(path: PathBuf, modules: &[EmittedModule]) -> Result<()> {
    write_lean_file(path, |w| {
        writeln!(w, "import {BUS_DEFS_IMPORT}")?;
        for m in modules {
            writeln!(w, "import {}", m.air_namespace)?;
        }
        Ok(())
    })
}

fn write_lean_file<P, FN>(path: P, write_fn: FN) -> Result<()>
where
    P: AsRef<Path>,
    FN: FnOnce(&mut BufWriter<fs::File>) -> Result<()>,
{
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("create Lean parent dir {}", parent.display()))?;
    }
    let file =
        fs::File::create(path).wrap_err_with(|| format!("create Lean file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    write_fn(&mut writer)?;
    writer
        .flush()
        .wrap_err_with(|| format!("flush Lean file {}", path.display()))?;
    Ok(())
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
