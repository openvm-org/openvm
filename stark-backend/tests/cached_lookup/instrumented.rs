use std::fs::{self, File};

use ax_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, prover::types::Proof, verifier::VerificationError,
};
use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2::{self, engine_from_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkEngineWithHashInstrumentation,
};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::prove::{get_data_sizes, prove, BenchParams};
use crate::{
    cached_lookup::prove::generate_random_trace,
    config::{
        instrument::{HashStatistics, StarkHashStatistics},
        FriParameters,
    },
};

fn instrumented_verify<SC: StarkGenericConfig, E: StarkEngineWithHashInstrumentation<SC>>(
    engine: &mut E,
    vk: MultiStarkVerifyingKey<SC>,
    air: &DummyInteractionAir,
    proof: Proof<SC>,
) -> StarkHashStatistics<BenchParams> {
    let degree = proof.per_air[0].degree;
    let log_degree = log2_ceil_usize(degree);

    engine.clear_instruments();
    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    // Do not check cumulative sum
    let res = verifier.verify(&mut challenger, &vk, &proof);
    if matches!(res, Err(ref err) if err != &VerificationError::NonZeroCumulativeSum) {
        panic!("{res:?}");
    };

    let bench_params = BenchParams {
        field_width: air.field_width(),
        log_degree,
    };
    engine.stark_hash_statistics(bench_params)
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
    let mut engine = engine_from_perm(instr_perm, fri_params);
    engine.perm.is_on = false;

    let (vk, air, proof, _) = prove(&engine, trace, partition);
    engine.perm.is_on = true;
    instrumented_verify(&mut engine, vk, &air, proof)
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
    let fri_params = [1, 2, 3, 4]
        .map(standard_fri_params_with_100_bits_conjectured_security)
        .to_vec();
    let data_sizes = get_data_sizes();

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
