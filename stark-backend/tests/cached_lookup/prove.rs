use std::{
    fs::{self, File},
    sync::Arc,
    time::Instant,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkVerifyingKey,
    prover::types::{Proof, ProofInput},
    utils::disable_debug_builder,
    Chip,
};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    dummy_airs::interaction::dummy_interaction_air::{
        DummyInteractionAir, DummyInteractionChip, DummyInteractionData,
    },
    engine::StarkEngine,
};
use p3_uni_stark::{Domain, StarkGenericConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// Lookup table is cached, everything else (including counts) is committed together
#[allow(clippy::type_complexity)]
pub fn prove<SC: StarkGenericConfig, E: StarkEngine<SC>>(
    engine: &E,
    trace: Vec<(u32, Vec<u32>)>,
    partition: bool,
) -> (
    MultiStarkVerifyingKey<SC>,
    Arc<DummyInteractionAir>,
    Proof<SC>,
    ProverBenchmarks,
)
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let mut chip =
        DummyInteractionChip::new_with_partition(engine.config().pcs(), trace[0].1.len(), false, 0);
    let (count, fields): (Vec<_>, Vec<_>) = trace.into_iter().unzip();
    let data = DummyInteractionData { count, fields };
    chip.load_data(data);

    let mut keygen_builder = engine.keygen_builder();
    let air_id = keygen_builder.add_air(chip.air());
    let pk = keygen_builder.generate_pk();
    let vk = pk.get_vk();

    let mut benchmarks = ProverBenchmarks::default();
    let prover = engine.prover();
    // Must add trace matrices in the same order as above
    let mut start;
    let air_proof_input = if partition {
        start = Instant::now();
        // Receiver fields table is cached
        let ret = chip.generate_air_proof_input_with_id(air_id);
        benchmarks.cached_commit_time = start.elapsed().as_micros();
        ret
    } else {
        chip.generate_air_proof_input_with_id(air_id)
    };
    let proof_input = ProofInput {
        per_air: vec![air_proof_input],
    };
    start = Instant::now();

    // Disable debug prover since we don't balance the buses
    disable_debug_builder();
    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, proof_input);
    benchmarks.prove_time_without_trace_gen = start.elapsed().as_micros();

    (vk, Arc::new(chip.air), proof, benchmarks)
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ProverBenchmarks {
    pub cached_commit_time: u128,
    /// Includes common main trace commitment time.
    pub prove_time_without_trace_gen: u128,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BenchParams {
    pub field_width: usize,
    pub log_degree: usize,
}

pub fn generate_random_trace(
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

pub fn get_data_sizes() -> Vec<(usize, usize)> {
    let format_data_sizes =
        |field_widths: &[usize], log_degrees: &[usize]| -> Vec<(usize, usize)> {
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
        format_data_sizes(&[1, 2, 5, 10, 50, 100], &[3, 5, 10, 13, 15, 16, 18, 20]);
    data_sizes.extend(format_data_sizes(&[200, 500, 1000], &[1, 2, 3, 5, 10]));
    data_sizes
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverStatistics {
    /// Identifier for the hash permutation
    pub name: String,
    pub fri_params: FriParameters,
    pub bench_params: BenchParams,
    pub without_ct: ProverBenchmarks,
    pub with_ct: ProverBenchmarks,
}

fn compare_provers(
    fri_params: FriParameters,
    field_width: usize,
    log_degree: usize,
) -> ProverStatistics {
    let rng = StdRng::seed_from_u64(0);
    let trace = generate_random_trace(rng, field_width, 1 << log_degree);
    let engine = engine_from_perm(random_perm(), log_degree, fri_params);
    let (_, _, _, without_ct) = prove(&engine, trace.clone(), false);

    let (_, _, _, with_ct) = prove(&engine, trace, true);

    ProverStatistics {
        name: "Poseidon2Perm16".to_string(),
        fri_params,
        bench_params: BenchParams {
            field_width,
            log_degree,
        },
        without_ct,
        with_ct,
    }
}

// Run with `RUSTFLAGS="-Ctarget-cpu=native" cargo t --release -- --ignored --nocapture bench_cached_trace_prover`
#[test]
#[ignore = "bench"]
fn bench_cached_trace_prover() -> eyre::Result<()> {
    let fri_params = [1, 2, 3, 4]
        .map(standard_fri_params_with_100_bits_conjectured_security)
        .to_vec();
    let data_sizes = get_data_sizes();

    // Write to csv as we go
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    let _ = fs::create_dir_all(format!("{}/data", cargo_manifest_dir));
    let csv_path = format!("{}/data/cached_trace_prover.csv", cargo_manifest_dir);
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
        "",
        "without_ct.main_commit_time(µs)",
        "without_ct.prove_time(µs)",
        "with_ct.cache_commit_time(µs)",
        "with_ct.main_commit_time(µs)",
        "with_ct.prove_time(µs)",
    ])?;

    let mut all_stats = vec![];
    for fri_param in fri_params {
        for (field_width, log_degree) in &data_sizes {
            let stats = compare_provers(fri_param, *field_width, *log_degree);
            wtr.serialize(&stats)?;
            wtr.flush()?;
            all_stats.push(stats);
        }
    }

    let json_path = format!("{}/data/cached_trace_prover.json", cargo_manifest_dir);
    let file = File::create(json_path)?;
    serde_json::to_writer(file, &all_stats)?;

    Ok(())
}
