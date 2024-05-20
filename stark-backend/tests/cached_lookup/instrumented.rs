use std::{
    fs::{self, File},
    iter,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkVerifyingKey,
    prover::{trace::TraceCommitmentBuilder, types::Proof},
};

use afs_test_utils::{
    config::baby_bear_poseidon2::{self, engine_from_perm},
    engine::{StarkEngine, StarkEngineWithHashInstrumentation},
    interaction::dummy_interaction_air::DummyInteractionAir,
};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::config::{
    instrument::{HashStatistics, StarkHashStatistics},
    FriParameters,
};

// type Val = BabyBear;

// Lookup table is cached, everything else (including counts) is committed together
#[allow(clippy::type_complexity)]
fn prove<SC: StarkGenericConfig, E: StarkEngine<SC>>(
    engine: &E,
    trace: Vec<(u32, Vec<u32>)>,
    partition: bool,
) -> (
    MultiStarkVerifyingKey<SC>,
    DummyInteractionAir,
    Proof<SC>,
    Vec<Vec<Val<SC>>>,
)
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    // tracing_setup();
    let degree = trace.len();

    let air = DummyInteractionAir::new(trace[0].1.len(), false, 0);

    // Single row major matrix for |count|fields[..]|
    let nopart_trace = RowMajorMatrix::new(
        trace
            .iter()
            .cloned()
            .flat_map(|(count, fields)| {
                assert_eq!(fields.len(), air.field_width());
                iter::once(count).chain(fields)
            })
            .map(Val::<SC>::from_wrapped_u32)
            .collect(),
        air.field_width() + 1,
    );
    let (count, fields): (Vec<_>, Vec<_>) = trace.into_iter().unzip();
    let part_count_trace = RowMajorMatrix::new(
        count.into_iter().map(Val::<SC>::from_wrapped_u32).collect(),
        1,
    );
    let part_fields_trace = RowMajorMatrix::new(
        fields
            .into_iter()
            .flat_map(|fields| {
                assert_eq!(fields.len(), air.field_width());
                fields
            })
            .map(Val::<SC>::from_wrapped_u32)
            .collect(),
        air.field_width(),
    );

    let mut keygen_builder = engine.keygen_builder();
    if partition {
        let fields_ptr = keygen_builder.add_cached_main_matrix(air.field_width());
        let count_ptr = keygen_builder.add_main_matrix(1);
        keygen_builder.add_partitioned_air(&air, degree, 0, vec![count_ptr, fields_ptr]);
    } else {
        keygen_builder.add_air(&air, degree, 0);
    }
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    // Must add trace matrices in the same order as above
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    if partition {
        // Receiver fields table is cached
        let cached_trace_data = trace_builder
            .committer
            .commit(vec![part_fields_trace.clone()]);
        trace_builder.load_cached_trace(part_fields_trace, cached_trace_data);
        trace_builder.load_trace(part_count_trace);
    } else {
        trace_builder.load_trace(nopart_trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);
    let pis = vec![vec![]];

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    (vk, air, proof, pis)
}

fn instrumented_verify<SC: StarkGenericConfig, E: StarkEngineWithHashInstrumentation<SC>>(
    engine: &mut E,
    vk: MultiStarkVerifyingKey<SC>,
    air: DummyInteractionAir,
    proof: Proof<SC>,
    pis: Vec<Vec<Val<SC>>>,
) -> StarkHashStatistics<BenchParams> {
    let degree = vk.per_air[0].degree;
    let log_degree = log2_ceil_usize(degree);

    engine.clear_instruments();
    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    // Do not check cumulative sum
    verifier
        .verify_raps(&mut challenger, vk, vec![&air], proof, &pis)
        .unwrap();

    let bench_params = BenchParams {
        field_width: air.field_width(),
        log_degree,
    };
    engine.stark_hash_statistics(bench_params)
}

fn generate_random_trace(
    mut rng: impl Rng,
    field_width: usize,
    height: usize,
) -> Vec<(u32, Vec<u32>)> {
    (0..height)
        .map(|_| {
            (
                rng.gen_range(0..1000),
                (0..field_width).map(|_| rng.gen()).collect(),
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BenchParams {
    pub field_width: usize,
    pub log_degree: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierStatistics {
    /// Identifier for the hash permutation
    pub name: String,
    pub fri_params: FriParameters,
    pub bench_params: BenchParams,
    pub without_ct: HashStatistics,
    pub with_ct: HashStatistics,
}

fn instrumented_prove_and_verify(
    fri_params: FriParameters,
    trace: Vec<(u32, Vec<u32>)>,
    partition: bool,
) -> StarkHashStatistics<BenchParams> {
    let instr_perm = baby_bear_poseidon2::random_instrumented_perm();
    let degree = trace.len();
    let pcs_log_degree = log2_ceil_usize(degree);
    let mut engine = engine_from_perm(instr_perm, pcs_log_degree, fri_params);
    engine.perm.is_on = false;

    let (vk, air, proof, pis) = prove(&engine, trace, partition);
    engine.perm.is_on = true;
    instrumented_verify(&mut engine, vk, air, proof, pis)
}

fn instrumented_verifier_comparison(
    fri_params: FriParameters,
    field_width: usize,
    log_degree: usize,
) -> VerifierStatistics {
    let rng = StdRng::seed_from_u64(0);
    let trace = generate_random_trace(rng, field_width, 1 << log_degree);
    println!("Without cached trace:");
    let without_ct = instrumented_prove_and_verify(fri_params, trace.clone(), false);

    println!("With cached trace:");
    let with_ct = instrumented_prove_and_verify(fri_params, trace, true);

    VerifierStatistics {
        name: without_ct.name,
        fri_params: without_ct.fri_params,
        bench_params: without_ct.custom,
        without_ct: without_ct.stats,
        with_ct: with_ct.stats,
    }
}

// Run with `RUSTFLAGS="-Ctarget-cpu=native" cargo t --release -- --ignored --nocapture instrument_cached_trace_verifier`
#[test]
#[ignore = "bench"]
fn instrument_cached_trace_verifier() -> eyre::Result<()> {
    let fri_params = [
        FriParameters {
            log_blowup: 1,
            num_queries: 100,
            proof_of_work_bits: 16,
        },
        FriParameters {
            log_blowup: 3,
            num_queries: 33,
            proof_of_work_bits: 16,
        },
        FriParameters {
            log_blowup: 4,
            num_queries: 45,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 4,
            num_queries: 69,
            proof_of_work_bits: 0,
        },
    ];
    let get_data_sizes = |field_widths: &[usize], log_degrees: &[usize]| -> Vec<(usize, usize)> {
        field_widths
            .iter()
            .flat_map(|field_width| {
                log_degrees
                    .iter()
                    .map(|log_degree| (*field_width, *log_degree))
            })
            .collect::<Vec<_>>()
    };
    let mut data_sizes: Vec<(usize, usize)> =
        get_data_sizes(&[1, 2, 5, 10, 50, 100], &[3, 5, 10, 13, 15, 16, 18, 20]);
    data_sizes.extend(get_data_sizes(&[200, 500, 1000], &[1, 2, 3, 5, 10]));

    // Write to csv as we go
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    let _ = fs::create_dir_all(format!("{}/data", cargo_manifest_dir));
    let csv_path = format!(
        "{}/data/cached_trace_instrumented_verifier.csv",
        cargo_manifest_dir
    );
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    // Manually write record because header cannot handle nested struct well
    wtr.write_record([
        "permutation_name",
        "log_blowup",
        "num_queries",
        "proof_of_work_bits",
        "page_width",
        "log_degree",
        "without_ct.permutations",
        "with_ct.permutations",
    ])?;

    let mut all_stats = vec![];
    for fri_param in fri_params {
        for (field_width, log_degree) in &data_sizes {
            let stats = instrumented_verifier_comparison(fri_param, *field_width, *log_degree);
            wtr.serialize(&stats)?;
            wtr.flush()?;
            all_stats.push(stats);
        }
    }

    let json_path = format!(
        "{}/data/cached_trace_instrumented_verifier.json",
        cargo_manifest_dir
    );
    let file = File::create(json_path)?;
    serde_json::to_writer(file, &all_stats)?;

    Ok(())
}
