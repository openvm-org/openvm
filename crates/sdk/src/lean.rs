//! Driver: walk the leaf aggregation circuit (recursion verifier
//! sub-circuit wrapped in `InnerCircuit`), render every supported AIR's
//! symbolic constraints, and emit per-AIR Lean modules in the
//! `Fundamentals.Air` dialect under
//! `<output>/Recursion/Airs/<Group>/<AirStem>/Generated/{Schema,Constraints,Interactions}.lean`.
//!
//! Unknown bus indices fall back to `bus_<idx>` placeholder constructors.

use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use eyre::{bail, Result, WrapErr};
use openvm_circuit::primitives::{ColumnsAir, StructReflectionHelper};
use openvm_continuations::circuit::{
    inner::{
        def_pvs::DeferralPvsAir, unset::UnsetPvsAir, verifier::VerifierPvsAir, vm_pvs::VmPvsAir,
    },
    Circuit,
};
use openvm_recursion_circuit::{
    batch_constraint::{
        eq_airs::{
            eq_3b::Eq3bAir,
            eq_neg::EqNegAir,
            eq_ns::EqNsAir,
            eq_sharp_uni::{EqSharpUniAir, EqSharpUniReceiverAir},
            eq_uni::EqUniAir,
        },
        expr_eval::{
            constraints_folding::ConstraintsFoldingAir,
            interactions_folding::InteractionsFoldingAir,
            symbolic_expression::SymbolicExpressionAir,
        },
        expression_claim::ExpressionClaimAir,
        fractions_folder::FractionsFolderAir,
        sumcheck::{multilinear::MultilinearSumcheckAir, univariate::UnivariateSumcheckAir},
    },
    gkr::{
        input::GkrInputAir, layer::GkrLayerAir, sumcheck::GkrLayerSumcheckAir,
        xi_sampler::GkrXiSamplerAir,
    },
    primitives::{exp_bits_len::ExpBitsLenAir, pow::PowerCheckerAir, range::RangeCheckerAir},
    proof_shape::{proof_shape::ProofShapeAir, pvs::PublicValuesAir},
    stacking::{
        claims::StackingClaimsAir, eq_base::EqBaseAir, eq_bits::EqBitsAir,
        opening::OpeningClaimsAir, sumcheck::SumcheckRoundsAir, univariate::UnivariateRoundAir,
    },
    transcript::{
        merkle_verify::MerkleVerifyAir, poseidon2::Poseidon2Air, transcript::TranscriptAir,
    },
    whir::{
        final_poly_mle_eval::FinalPolyMleEvalAir, final_poly_query_eval::FinalPolyQueryEvalAir,
        folding::WhirFoldingAir, initial_opened_values::InitialOpenedValuesAir,
        non_initial_opened_values::NonInitialOpenedValuesAir, query::WhirQueryAir,
        sumcheck::SumcheckAir, whir_round::WhirRoundAir,
    },
};
use openvm_stark_backend::{
    interaction::BusIndex,
    lean::{
        air_file_stem, render_air, write_constraints, write_interactions, write_schema, BusBinding,
        LeanRenderOptions, LeanWriteOptions, RenderedAir,
    },
    AirRef,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs};

use crate::{config, Sdk, SC};

const ROOT_MODULE: &str = "Recursion";
const AIRS_MODULE: &str = "Airs";
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
        "transcriptBus",               // 0
        "poseidon2PermuteBus",         // 1
        "poseidon2CompressBus",        // 2
        "merkleVerifyBus",             // 3
        "gkrModuleBus",                // 4
        "batchConstraintModuleBus",    // 5
        "stackingModuleBus",           // 6
        "whirModuleBus",               // 7
        "whirMuBus",                   // 8
        "airShapeBus",                 // 9
        "airPresenceBus",              // 10
        "hyperdimBus",                 // 11
        "liftedHeightsBus",            // 12
        "stackingIndicesBus",          // 13
        "commitmentsBus",              // 14
        "publicValuesBus",             // 15
        "selUniBus",                   // 16
        "rangeCheckerBus",             // 17
        "powerCheckerBus",             // 18
        "expressionClaimNMaxBus",      // 19
        "constraintsFoldingInputBus",  // 20
        "interactionsFoldingInputBus", // 21
        "fractionFolderInputBus",      // 22
        "nLiftBus",                    // 23
        "eqNLogupNMaxBus",             // 24
        "eq3bShapeBus",                // 25
        "xiRandomnessBus",             // 26
        "constraintRandomnessBus",     // 27
        "whirOpeningPointBus",         // 28
        "whirOpeningPointLookupBus",   // 29
        "columnClaimsBus",             // 30
        "expBitsLenBus",               // 31
        "rightShiftBus",               // 32
        "eqNegBaseRandBus",            // 33
        "eqNegResultBus",              // 34
        "cachedCommitBus",             // 35
        "preHashBus",                  // 36
        "finalStateBus",               // 37
    ];
    // Per-module internal buses, in module instantiation order:
    // transcript (0 buses), proof_shape (3), gkr (6), batch_constraint
    // (13), stacking (8), whir (12). Total: 38 + 0 + 3 + 6 + 13 + 8 + 12 = 80.
    let module_internal_offset = 38;
    let module_internal_names: &[&str] = &[
        // ProofShapeModule (38–40)
        "proofShapePermutationBus", // 38
        "startingTidxBus",          // 39
        "numPvsBus",                // 40
        // GkrModule (41–46)
        "gkrLayerInputBus",        // 41
        "gkrLayerOutputBus",       // 42
        "gkrSumcheckInputBus",     // 43
        "gkrSumcheckOutputBus",    // 44
        "gkrSumcheckChallengeBus", // 45
        "gkrXiSamplerBus",         // 46
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
        "stackingTidxBus",      // 60
        "claimCoefficientsBus", // 61
        "sumcheckClaimsBus",    // 62
        "eqRandValuesBus",      // 63
        "eqBaseBus",            // 64
        "eqBitsInternalBus",    // 65
        "eqKernelLookupBus",    // 66
        "eqBitsLookupBus",      // 67
        // WhirModule (68–79)
        "whirSumcheckBus",       // 68
        "whirAlphaBus",          // 69
        "whirGammaBus",          // 70
        "whirQueryBus",          // 71
        "verifyQueriesBus",      // 72
        "verifyQueryBus",        // 73
        "whirEqAlphaUBus",       // 74
        "whirFoldingBus",        // 75
        "finalPolyMleEvalBus",   // 76
        "finalPolyQueryEvalBus", // 77
        "whirFinalPolyBus",      // 78
        "finalPolyFoldingBus",   // 79
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
    // Buses added by `InnerCircuit` on top of the verifier sub-circuit
    // (allocated at `next_bus_idx`, i.e. starting at 80).
    let leaf_bus_offset = module_internal_offset + module_internal_names.len();
    let leaf_bus_names: &[&str] = &[
        "pvsAirConsistencyBus", // 80
    ];
    bindings.extend(leaf_bus_names.iter().enumerate().map(|(i, n)| BusBinding {
        vk_index: (leaf_bus_offset + i) as BusIndex,
        lean_name: (*n).to_string(),
    }));
    bindings
}

/// Group classification: which subdirectory of `Recursion/` the AIR
/// goes into. Determines its Lean namespace.
fn air_group(air_name: &str) -> Result<&'static str> {
    Ok(match air_name {
        n if n.starts_with("DeferralPvsAir")
            || n.starts_with("UnsetPvsAir")
            || n.starts_with("VerifierPvsAir")
            || n.starts_with("VmPvsAir") =>
        {
            "Continuations"
        }
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

fn columns_from_air_trait(air: &AirRef<SC>) -> Option<Vec<String>> {
    macro_rules! try_air {
        ($ty:ty) => {
            if let Some(air) = air.as_any().downcast_ref::<$ty>() {
                if let Some(columns) = air.columns() {
                    return Some(normalize_columns_air_names(columns));
                }
            }
        };
    }

    try_air!(VerifierPvsAir);
    try_air!(VmPvsAir);
    try_air!(UnsetPvsAir);
    try_air!(DeferralPvsAir);

    try_air!(FractionsFolderAir);
    try_air!(UnivariateSumcheckAir);
    try_air!(MultilinearSumcheckAir);
    try_air!(EqNsAir);
    try_air!(Eq3bAir);
    try_air!(EqSharpUniReceiverAir);
    try_air!(EqSharpUniAir);
    try_air!(EqUniAir);
    try_air!(ExpressionClaimAir);
    try_air!(InteractionsFoldingAir);
    try_air!(ConstraintsFoldingAir);
    try_air!(SymbolicExpressionAir<F>);
    try_air!(EqNegAir);
    try_air!(TranscriptAir);
    try_air!(MerkleVerifyAir);
    try_air!(Poseidon2Air<F, 1>);
    try_air!(PublicValuesAir);
    try_air!(ProofShapeAir<4, 8>);
    try_air!(RangeCheckerAir<8>);
    try_air!(GkrInputAir);
    try_air!(GkrLayerSumcheckAir);
    try_air!(GkrLayerAir);
    try_air!(GkrXiSamplerAir);
    try_air!(OpeningClaimsAir);
    try_air!(UnivariateRoundAir);
    try_air!(SumcheckRoundsAir);
    try_air!(StackingClaimsAir);
    try_air!(EqBaseAir);
    try_air!(EqBitsAir);
    try_air!(SumcheckAir);
    try_air!(WhirQueryAir);
    try_air!(InitialOpenedValuesAir);
    try_air!(NonInitialOpenedValuesAir);
    try_air!(WhirFoldingAir);
    try_air!(FinalPolyMleEvalAir);
    try_air!(FinalPolyQueryEvalAir);
    try_air!(WhirRoundAir);
    try_air!(PowerCheckerAir<2, 32>);
    try_air!(ExpBitsLenAir);

    None
}

fn normalize_columns_air_names(columns: Vec<String>) -> Vec<String> {
    columns
        .into_iter()
        .map(|mut name| {
            while name.contains("__") {
                name = name.replace("__", "_");
            }
            if name.ends_with("_aux_inv") {
                name.truncate(name.len() - "_inv".len());
                name.push_str("_0");
            }
            name
        })
        .collect()
}

fn flat_column_names(air: &AirRef<SC>) -> Option<Vec<String>> {
    columns_from_air_trait(air)
}

/// Names for the AIR's flat public-value list, in the order matching
/// the AIR's `BaseAirWithPublicValues::num_public_values()` flattening.
/// Returned slice is empty for AIRs without public values.
fn flat_pv_names(air_name: &str) -> Vec<String> {
    if air_name.starts_with("VerifierPvsAir") {
        normalize_columns_air_names(
            VerifierBasePvs::<u8>::struct_reflection().expect("VerifierBasePvs column reflection"),
        )
    } else if air_name.starts_with("VmPvsAir") {
        normalize_columns_air_names(
            VmPvs::<u8>::struct_reflection().expect("VmPvs column reflection"),
        )
    } else {
        Vec::new()
    }
}

/// Generates Lean files for the leaf aggregation circuit AIRs under a
/// `Recursion` tree at `output_dir`. The leaf circuit is the recursion
/// verifier sub-circuit wrapped in `InnerCircuit` (with deferral
/// disabled, matching `Sdk::standard`'s leaf prover keygen).
pub fn generate_lean_files_for_leaf_circuit<P: AsRef<Path>>(output_dir: P) -> Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).wrap_err("create Lean output dir")?;

    let recursion_dir = output_dir.join(ROOT_MODULE);
    fs::create_dir_all(&recursion_dir).wrap_err("create Recursion output dir")?;
    let airs_dir = recursion_dir.join(AIRS_MODULE);
    fs::create_dir_all(&airs_dir).wrap_err("create Recursion/Airs output dir")?;

    eprintln!("[lean] Building Sdk::standard...");
    let agg_params = config::AggregationSystemParams::default();
    let app_params = openvm_stark_sdk::config::app_params_with_100_bits_security(
        openvm_stark_sdk::config::MAX_APP_LOG_STACKED_HEIGHT,
    );
    let sdk = Sdk::standard(app_params, agg_params);

    eprintln!("[lean] Running leaf prover keygen via SDK...");
    let agg_prover = sdk.agg_prover();
    let leaf_circuit = agg_prover.leaf_prover.get_circuit();
    let vk = agg_prover.leaf_prover.get_vk();
    let airs = Circuit::<SC>::airs(&*leaf_circuit);
    eprintln!("[lean] Leaf circuit has {} AIRs", airs.len());

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
        // layout (cached at part 0, common singles at part 1).
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

        let column_names = match flat_column_names(air) {
            Some(n) => n,
            None => {
                skipped.push((air_name.clone(), "unsupported AIR shape (v1)"));
                continue;
            }
        };
        let expected_width: usize =
            cached_widths.iter().sum::<usize>() + air_vk.params.width.common_main;
        if column_names.len() != expected_width {
            skipped.push((air_name.clone(), "column count mismatch with main_width"));
            continue;
        }

        let stem = air_file_stem(&air_name).to_string();
        let air_namespace = format!("{ROOT_MODULE}.{group}.{stem}");
        let air_import_module = format!("{ROOT_MODULE}.{AIRS_MODULE}.{group}.{stem}");
        let schema_import = format!("{air_import_module}.Generated.Schema");
        let constraints_import = format!("{air_import_module}.Generated.Constraints");
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
            public_value_names: flat_pv_names(&air_name),
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

        let stem_dir = airs_dir.join(group).join(&stem);
        let generated_dir = stem_dir.join("Generated");
        fs::create_dir_all(&generated_dir)
            .wrap_err_with(|| format!("create dir {}", generated_dir.display()))?;

        write_lean_file(generated_dir.join("Schema.lean"), |w| {
            write_schema(w, &air_name, &column_names, &opts).wrap_err("write Schema.lean")?;
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
            air_import_module,
        });
    }

    write_per_air_module_files(&airs_dir, &emitted)?;
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
    air_import_module: String,
}

/// Per-AIR module file `<Group>/<AirStem>.lean`: imports its own five
/// sub-files (Schema, Constraints, Interactions, Labels, Facts).
fn write_per_air_module_files(airs_dir: &Path, modules: &[EmittedModule]) -> Result<()> {
    for m in modules {
        let path = airs_dir.join(m.group).join(format!("{}.lean", m.stem));
        let air_import_module = &m.air_import_module;
        write_lean_file(path, |w| {
            writeln!(w, "import {air_import_module}.Generated.Schema")?;
            writeln!(w, "import {air_import_module}.Generated.Constraints")?;
            writeln!(w, "import {air_import_module}.Generated.Interactions")?;
            writeln!(w, "import {air_import_module}.Labels")?;
            writeln!(w, "import {air_import_module}.Facts")?;
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
            writeln!(w, "import {}", m.air_import_module)?;
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
    fn generates_lean_files_for_leaf_circuit() {
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lean_out");
        generate_lean_files_for_leaf_circuit(output_dir).unwrap();
    }
}
