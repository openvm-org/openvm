use std::{
    collections::{BTreeSet, HashSet},
    fs,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use eyre::{bail, eyre, Result, WrapErr};
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
            DagCommitCols,
            symbolic_expression::{
                CachedSymbolicExpressionColumns, SingleMainSymbolicExpressionColumns,
                SymbolicExpressionAir,
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
    proof_shape::{
        proof_shape::{ProofShapeAir, ProofShapeCols},
        pvs::PublicValuesCols,
    },
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
        bus_defs_to_lean_writer, constraints_scaffold_to_lean_writer,
        extract_constraints_dag_to_lean_writer_with_options,
        extraction_intermediate_attrs_to_lean_writer, generate_lean_air_definition,
        LeanColumns, LeanConstraintsScaffoldOptions, LeanEntry, LeanExtractionOptions,
    },
    AirRef, StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DuplexSponge, DIGEST_SIZE};

use crate::{config, F, Sdk, SC};

const MAX_NUM_PROOFS: usize = 4;
const ROOT_MODULE: &str = "Recursion";
const EXTRACTION_ATTRS_MODULE: &str = "Recursion.Extraction.Attributes";
const BUS_DEFS_MODULE: &str = "Recursion.BusDefs";
const BUS_DEFS_NAMESPACE: &str = "Recursion.BusDefs";

#[derive(Clone, Debug)]
struct LeanModuleInfo {
    air_name: String,
    file_stem: String,
    group: &'static str,
    air_module: String,
    extraction_module: String,
    constraints_module: String,
}

impl LeanModuleInfo {
    fn from_air_name(air_name: &str) -> Result<Self> {
        let group = air_group(air_name)?;
        let file_stem = air_file_stem(air_name).to_string();
        Ok(Self {
            air_name: air_name.to_string(),
            air_module: format!("{ROOT_MODULE}.Airs.{group}.{file_stem}"),
            extraction_module: format!("{ROOT_MODULE}.Extraction.{group}.{file_stem}"),
            constraints_module: format!("{ROOT_MODULE}.Constraints.{group}.{file_stem}"),
            file_stem,
            group,
        })
    }

    fn air_path(&self, root: &Path) -> PathBuf {
        root.join("Airs")
            .join(self.group)
            .join(format!("{}.lean", self.file_stem))
    }

    fn extraction_path(&self, root: &Path) -> PathBuf {
        root.join("Extraction")
            .join(self.group)
            .join(format!("{}.lean", self.file_stem))
    }

    fn constraints_path(&self, root: &Path) -> PathBuf {
        root.join("Constraints")
            .join(self.group)
            .join(format!("{}.lean", self.file_stem))
    }
}

/// Generates Lean files for the verifier recursion circuit AIRs under a `Recursion` tree.
pub fn generate_lean_files_for_recursion_circuit<P: AsRef<Path>>(output_dir: P) -> Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).wrap_err("create Lean output dir")?;

    let recursion_dir = output_dir.join(ROOT_MODULE);
    fs::create_dir_all(&recursion_dir).wrap_err("create Recursion output dir")?;

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

    let module_infos = airs
        .iter()
        .map(|air| LeanModuleInfo::from_air_name(&air.name()))
        .collect::<Result<Vec<_>>>()?;
    let symbolic_constraints = vk
        .inner
        .per_air
        .iter()
        .map(|air_vk| &air_vk.symbolic_constraints)
        .collect::<Vec<_>>();

    eprintln!("[lean_extract] Writing shared Lean files...");
    write_lean_file(recursion_dir.join("Extraction/Attributes.lean"), |writer| {
        extraction_intermediate_attrs_to_lean_writer(
            module_infos.iter().map(|info| info.air_name.as_str()),
            writer,
        )
        .wrap_err("write extraction attrs")?;
        Ok(())
    })?;
    write_lean_file(recursion_dir.join("BusDefs.lean"), |writer| {
        bus_defs_to_lean_writer(
            symbolic_constraints.iter().copied(),
            BUS_DEFS_NAMESPACE,
            writer,
        )
        .wrap_err("write bus defs")?;
        Ok(())
    })?;

    eprintln!("[lean_extract] Writing per-air Lean files...");
    for (i, ((air, air_vk), module_info)) in airs
        .iter()
        .zip(vk.inner.per_air.iter())
        .zip(module_infos.iter())
        .enumerate()
    {
        let air_name = &module_info.air_name;
        let main_width = air_vk.params.width.main_width();
        let t_air = std::time::Instant::now();

        write_air_file(&module_info.air_path(&recursion_dir), i, air, air_name, main_width)
            .wrap_err_with(|| format!("write AIR file for {air_name}"))?;
        write_extraction_file(
            &module_info.extraction_path(&recursion_dir),
            i,
            air_name,
            main_width,
            &air_vk.symbolic_constraints,
        )
        .wrap_err_with(|| format!("write extraction file for {air_name}"))?;
        write_constraints_file(
            &module_info.constraints_path(&recursion_dir),
            air_name,
            module_info,
            &air_vk.symbolic_constraints,
        )
        .wrap_err_with(|| format!("write constraints scaffold for {air_name}"))?;

        eprintln!(
            "[lean_extract]   [{i:02}] {air_name} — total {:?}",
            t_air.elapsed(),
        );
    }

    write_root_module(output_dir.join(format!("{ROOT_MODULE}.lean")), &module_infos)
        .wrap_err("write root Recursion module")?;

    eprintln!(
        "[lean_extract] Done. Wrote Lean tree to {} (total {:?})",
        output_dir.display(),
        t0.elapsed(),
    );

    Ok(())
}

fn write_air_file(
    path: &Path,
    index: usize,
    air: &AirRef<SC>,
    air_name: &str,
    main_width: usize,
) -> Result<()> {
    let body = render_air_definition(air, air_name)?;
    write_lean_file(path, |writer| {
        write_air_header(writer, index, air_name, main_width).wrap_err("write AIR header")?;
        writeln!(writer, "import LeanZKCircuit.Command.Air.define_air")
            .wrap_err("write AIR import")?;
        writeln!(writer, "import LeanZKCircuit.OpenVM.Circuit").wrap_err("write AIR import")?;
        writeln!(writer).wrap_err("write AIR spacing")?;
        writeln!(writer, "set_option linter.unusedVariables false")
            .wrap_err("write AIR option")?;
        if needs_max_rec_depth(air_name) {
            writeln!(writer, "set_option maxRecDepth 1024").wrap_err("write AIR option")?;
        }
        writeln!(writer).wrap_err("write AIR spacing")?;
        writeln!(writer, "{body}").wrap_err("write AIR body")?;
        Ok(())
    })
}

fn write_extraction_file(
    path: &Path,
    index: usize,
    air_name: &str,
    main_width: usize,
    symbolic_constraints: &openvm_stark_backend::air_builders::symbolic::SymbolicConstraintsDag<F>,
) -> Result<()> {
    let helper_attr = format!(
        "{}_extraction_intermediates",
        format_lean_air_name(air_name)
    );
    let options = LeanExtractionOptions {
        attrs_import: Some(EXTRACTION_ATTRS_MODULE),
        helper_attr: Some(&helper_attr),
        register_inline_attrs: true,
    };

    write_lean_file(path, |writer| {
        write_air_header(writer, index, air_name, main_width)
            .wrap_err("write extraction header")?;
        extract_constraints_dag_to_lean_writer_with_options(
            symbolic_constraints,
            air_name,
            &options,
            writer,
        )
        .wrap_err("write extraction body")?;
        Ok(())
    })
}

fn write_constraints_file(
    path: &Path,
    air_name: &str,
    module_info: &LeanModuleInfo,
    symbolic_constraints: &openvm_stark_backend::air_builders::symbolic::SymbolicConstraintsDag<F>,
) -> Result<()> {
    let air_definition_name = air_definition_name_for(air_name);
    let options = LeanConstraintsScaffoldOptions {
        air_import: &module_info.air_module,
        air_definition_name: air_definition_name.as_deref(),
        attrs_import: EXTRACTION_ATTRS_MODULE,
        extraction_import: &module_info.extraction_module,
        bus_defs_import: BUS_DEFS_MODULE,
        bus_defs_namespace: BUS_DEFS_NAMESPACE,
    };

    write_lean_file(path, |writer| {
        constraints_scaffold_to_lean_writer(symbolic_constraints, air_name, &options, writer)
            .wrap_err("write constraints scaffold")?;
        Ok(())
    })
}

fn write_root_module(path: PathBuf, module_infos: &[LeanModuleInfo]) -> Result<()> {
    let mut imports = BTreeSet::new();
    let mut commented_imports = BTreeSet::new();
    imports.insert(BUS_DEFS_MODULE.to_string());
    imports.insert(EXTRACTION_ATTRS_MODULE.to_string());
    for info in module_infos {
        imports.insert(info.air_module.clone());
        imports.insert(info.extraction_module.clone());
        commented_imports.insert(info.constraints_module.clone());
    }

    write_lean_file(path, |writer| {
        for import in imports {
            writeln!(writer, "import {import}").wrap_err("write root import")?;
        }
        for import in commented_imports {
            writeln!(writer, "-- import {import}").wrap_err("write commented root import")?;
        }
        Ok(())
    })
}

fn write_lean_file<P, F>(path: P, write_fn: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufWriter<fs::File>) -> Result<()>,
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

fn write_air_header<W: Write>(
    writer: &mut W,
    index: usize,
    air_name: &str,
    main_width: usize,
) -> io::Result<()> {
    writeln!(writer, "-- AIR {index}: {air_name}")?;
    writeln!(writer, "-- main_width: {main_width}")?;
    writeln!(writer)?;
    Ok(())
}

fn render_air_definition(air: &AirRef<SC>, air_name: &str) -> Result<String> {
    let lean_air_name = format_lean_air_name(air_name);
    let definition = match air_name {
        n if n.starts_with("SymbolicExpressionAir") => {
            let air = air
                .as_any()
                .downcast_ref::<SymbolicExpressionAir<F>>()
                .ok_or_else(|| eyre!("expected SymbolicExpressionAir for {air_name}"))?;
            render_symbolic_expression_air_definition(air, air_name)
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
            let air = air
                .as_any()
                .downcast_ref::<ProofShapeAir<4, 8>>()
                .ok_or_else(|| eyre!("expected ProofShapeAir<4, 8> for {air_name}"))?;
            render_proof_shape_air_definition(air, air_name)
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
            let enc1 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 1>>(
                "WhirRoundAir_enc1",
            );
            let enc2 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 2>>(
                "WhirRoundAir_enc2",
            );
            let enc3 = generate_lean_air_definition_with_subairs::<WhirRoundCols<u8, 3>>(
                "WhirRoundAir_enc3",
            );
            format!("-- WhirRoundAir: ENC_WIDTH depends on num_whir_rounds encoder\n{enc1}\n\n{enc2}\n\n{enc3}")
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
        _ => bail!("unknown LeanColumns mapping for {air_name}"),
    };
    Ok(rewrite_reserved_lean_column_names_in_definition(definition))
}

fn render_symbolic_expression_air_definition(
    air: &SymbolicExpressionAir<F>,
    air_name: &str,
) -> String {
    let define_air_name =
        air_definition_name_for(air_name).unwrap_or_else(|| format_lean_air_name(air_name));
    let single_main_name = format!("{define_air_name}_SingleMain");
    let main_name = format!("{define_air_name}_Main");
    let single_main_width = lean_width_of::<SingleMainSymbolicExpressionColumns<u8>>();

    let mut sections = Vec::new();
    if air.dag_commit_subair.is_none() {
        let cached_name = format!("{define_air_name}_Cached");
        let cached_width = lean_width_of::<CachedSymbolicExpressionColumns<u8>>();
        sections.push(format!(
            "-- Atypical AIR: width = Cached + SingleMain * {}\n-- Cached partition (has_cached=true):\n{}",
            air.cnt_proofs,
            generate_lean_subair_definition::<CachedSymbolicExpressionColumns<u8>>(&cached_name),
        ));
        sections.push(format!(
            "-- Single main partition (repeated per proof):\n{}",
            generate_lean_subair_definition::<SingleMainSymbolicExpressionColumns<u8>>(
                &single_main_name,
            ),
        ));
        sections.push(format!(
            "-- Main partition: MAX_NUM_PROOFS copies of SingleMain\n{}",
            render_lean_definition(
                &main_name,
                LeanDefinitionKind::SubAir,
                (0..air.cnt_proofs)
                    .map(|i| LeanEntry::SubAir {
                        field_name: format!("single_main_{i}"),
                        type_name: single_main_name.clone(),
                        width: single_main_width,
                    })
                    .collect(),
            ),
        ));
        sections.push(format!(
            "-- Full AIR: cached partition plus repeated main partition\n{}",
            render_lean_definition(
                &define_air_name,
                LeanDefinitionKind::Air,
                vec![
                    LeanEntry::SubAir {
                        field_name: "cached".to_string(),
                        type_name: cached_name,
                        width: cached_width,
                    },
                    LeanEntry::SubAir {
                        field_name: "main_cols".to_string(),
                        type_name: main_name,
                        width: single_main_width * air.cnt_proofs,
                    },
                ],
            ),
        ));
    } else {
        let commit_name = format!("{define_air_name}_Commit");
        let commit_width = lean_width_of::<DagCommitCols<u8>>();
        sections.push(format!(
            "-- Atypical AIR: width = Commit + SingleMain * {}\n-- Commit partition (dag_commit_subair present):\n{}",
            air.cnt_proofs,
            generate_lean_subair_definition::<DagCommitCols<u8>>(&commit_name),
        ));
        sections.push(format!(
            "-- Single main partition (repeated per proof):\n{}",
            generate_lean_subair_definition::<SingleMainSymbolicExpressionColumns<u8>>(
                &single_main_name,
            ),
        ));
        sections.push(format!(
            "-- Main partition: MAX_NUM_PROOFS copies of SingleMain\n{}",
            render_lean_definition(
                &main_name,
                LeanDefinitionKind::SubAir,
                (0..air.cnt_proofs)
                    .map(|i| LeanEntry::SubAir {
                        field_name: format!("single_main_{i}"),
                        type_name: single_main_name.clone(),
                        width: single_main_width,
                    })
                    .collect(),
            ),
        ));
        sections.push(format!(
            "-- Full AIR: commit partition plus repeated main partition\n{}",
            render_lean_definition(
                &define_air_name,
                LeanDefinitionKind::Air,
                vec![
                    LeanEntry::SubAir {
                        field_name: "commit".to_string(),
                        type_name: commit_name,
                        width: commit_width,
                    },
                    LeanEntry::SubAir {
                        field_name: "main_cols".to_string(),
                        type_name: main_name,
                        width: single_main_width * air.cnt_proofs,
                    },
                ],
            ),
        ));
    }

    sections.join("\n\n")
}

fn render_proof_shape_air_definition(air: &ProofShapeAir<4, 8>, air_name: &str) -> String {
    let lean_air_name = format_lean_air_name(air_name);
    let fixed_name = format!("{lean_air_name}_Fixed");
    let encoder_name = format!("{lean_air_name}_Encoder");
    let cached_name = format!("{lean_air_name}_Cached");
    let fixed_width = lean_width_of::<ProofShapeCols<u8, 4>>();
    let encoder_width = air.idx_encoder.width();
    let cached_width = air.max_cached * DIGEST_SIZE;

    let mut sections = vec![format!(
        "-- Atypical AIR: width = ProofShapeCols + idx_encoder.width() + max_cached * DIGEST_SIZE\n-- Fixed portion (ProofShapeCols<NUM_LIMBS=4>):\n{}",
        generate_lean_subair_definition::<ProofShapeCols<u8, 4>>(&fixed_name),
    )];
    sections.push(format!(
        "-- Encoder portion (idx_encoder.width() = {encoder_width}):\n{}",
        render_lean_definition(
            &encoder_name,
            LeanDefinitionKind::SubAir,
            (0..encoder_width)
                .map(|i| LeanEntry::Column(format!("idx_flags_{i}")))
                .collect(),
        ),
    ));

    if cached_width > 0 {
        sections.push(format!(
            "-- Cached commitments portion (max_cached = {}, DIGEST_SIZE = {DIGEST_SIZE}):\n{}",
            air.max_cached,
            render_lean_definition(
                &cached_name,
                LeanDefinitionKind::SubAir,
                (0..air.max_cached)
                    .flat_map(|cached_idx| {
                        (0..DIGEST_SIZE).map(move |elem_idx| {
                            LeanEntry::Column(format!("cached_commits_{cached_idx}_{elem_idx}"))
                        })
                    })
                    .collect(),
            ),
        ));
    }

    let mut final_entries = vec![
        LeanEntry::SubAir {
            field_name: "fixed".to_string(),
            type_name: fixed_name,
            width: fixed_width,
        },
        LeanEntry::SubAir {
            field_name: "encoder".to_string(),
            type_name: encoder_name,
            width: encoder_width,
        },
    ];
    if cached_width > 0 {
        final_entries.push(LeanEntry::SubAir {
            field_name: "cached".to_string(),
            type_name: cached_name,
            width: cached_width,
        });
    }
    sections.push(format!(
        "-- Full AIR: fixed columns + encoder columns{}\n{}",
        if cached_width > 0 {
            " + cached commitment columns"
        } else {
            ""
        },
        render_lean_definition(&lean_air_name, LeanDefinitionKind::Air, final_entries),
    ));

    sections.join("\n\n")
}

fn air_group(air_name: &str) -> Result<&'static str> {
    Ok(match air_name {
        n if matches!(
            n,
            _
                if n.starts_with("ConstraintsFoldingAir")
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
                    || n.starts_with("UnivariateSumcheckAir")
        ) => "BatchConstraint",
        n if matches!(
            n,
            _
                if n.starts_with("GkrInputAir")
                    || n.starts_with("GkrLayerAir")
                    || n.starts_with("GkrLayerSumcheckAir")
                    || n.starts_with("GkrXiSamplerAir")
        ) => "GKR",
        n if matches!(
            n,
            _ if n.starts_with("ExpBitsLenAir")
                || n.starts_with("PowerCheckerAir")
                || n.starts_with("RangeCheckerAir")
        ) => "Primitive",
        n if matches!(n, _ if n.starts_with("ProofShapeAir") || n.starts_with("PublicValuesAir")) => {
            "ProofShape"
        }
        n if matches!(
            n,
            _
                if n.starts_with("EqBaseAir")
                    || n.starts_with("EqBitsAir")
                    || n.starts_with("OpeningClaimsAir")
                    || n.starts_with("StackingClaimsAir")
                    || n.starts_with("SumcheckRoundsAir")
                    || n.starts_with("UnivariateRoundAir")
        ) => "Stacking",
        n if matches!(
            n,
            _
                if n.starts_with("MerkleVerifyAir")
                    || n.starts_with("Poseidon2Air")
                    || n.starts_with("TranscriptAir")
        ) => "Transcript",
        n if matches!(
            n,
            _
                if n.starts_with("FinalPolyMleEvalAir")
                    || n.starts_with("Finaly")
                    || n.starts_with("FinalPolyQueryEvalAir")
                    || n.starts_with("InitialOpenedValuesAir")
                    || n.starts_with("NonInitialOpenedValuesAir")
                    || n.starts_with("SumcheckAir")
                    || n.starts_with("WhirFoldingAir")
                    || n.starts_with("WhirQueryAir")
                    || n.starts_with("WhirRoundAir")
        ) => "WHIR",
        _ => bail!("unknown recursion AIR group for {air_name}"),
    })
}

fn air_file_stem(air_name: &str) -> &str {
    air_name.split('<').next().unwrap_or(air_name)
}

fn air_definition_name_for(air_name: &str) -> Option<String> {
    if air_name.starts_with("SymbolicExpressionAir") {
        Some(air_file_stem(air_name).to_string())
    } else {
        None
    }
}

fn needs_max_rec_depth(air_name: &str) -> bool {
    air_name.starts_with("Poseidon2Air")
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

#[derive(Copy, Clone)]
enum LeanDefinitionKind {
    Air,
    SubAir,
}

fn render_lean_definition(
    air_name: &str,
    kind: LeanDefinitionKind,
    entries: Vec<LeanEntry>,
) -> String {
    let (directive, nested_directive) = match kind {
        LeanDefinitionKind::Air => ("#define_air", "MainSubAir"),
        LeanDefinitionKind::SubAir => ("#define_subair", "SubAir"),
    };

    let mut lines = vec![format!(
        "{directive} \"{air_name}\" using \"openvm_encapsulation\" where"
    )];
    for entry in entries {
        match entry {
            LeanEntry::Column(name) => lines.push(format!(
                "  Column[\"{}\"]",
                rewrite_reserved_lean_column_name(&name)
            )),
            LeanEntry::SubAir {
                field_name,
                type_name,
                width,
            } => lines.push(format!(
                "  {nested_directive}[\"{field_name}\": \"{type_name}\" width := {width}]"
            )),
        }
    }
    lines.join("\n")
}

fn rewrite_reserved_lean_column_name(name: &str) -> String {
    match name {
        "eq" => "eq_val".to_string(),
        _ => match name.strip_prefix("eq_") {
            Some(suffix) if suffix.chars().all(|ch| ch.is_ascii_digit()) => {
                format!("eq_val_{suffix}")
            }
            _ => name.to_string(),
        },
    }
}

fn rewrite_reserved_lean_column_names_in_definition(definition: String) -> String {
    definition
        .lines()
        .map(|line| {
            let Some(name) = line
                .strip_prefix("  Column[\"")
                .and_then(|rest| rest.strip_suffix("\"]"))
            else {
                return line.to_string();
            };
            format!("  Column[\"{}\"]", rewrite_reserved_lean_column_name(name))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_lean_subair_definition<C: LeanColumns>(air_name: &str) -> String {
    render_lean_definition(air_name, LeanDefinitionKind::SubAir, C::lean_columns())
}

fn lean_width_of<C: LeanColumns>() -> usize {
    C::lean_columns()
        .into_iter()
        .map(|entry| match entry {
            LeanEntry::Column(_) => 1,
            LeanEntry::SubAir { width, .. } => width,
        })
        .sum()
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
