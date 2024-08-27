use afs_compiler::util::execute_program;
use afs_stark_backend::{
    air_builders::PartitionedAirBuilder, prover::trace::TraceCommitmentBuilder,
    verifier::VerificationError,
};
use afs_test_utils::{
    config::baby_bear_poseidon2::default_engine, engine::StarkEngine, utils::generate_random_matrix,
};
use common::VerificationParams;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};

mod common;

/// Inner value is width of y-submatrix
pub struct SumAir(pub usize);

impl<F> BaseAir<F> for SumAir {
    fn width(&self) -> usize {
        self.0 + 1
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for SumAir {
    fn eval(&self, builder: &mut AB) {
        let partitioned_main = builder.partitioned_main();
        assert_eq!(partitioned_main.len(), 2);

        let x = partitioned_main[0].row_slice(0)[0];
        let ys = partitioned_main[1].row_slice(0);

        let mut y_sum = AB::Expr::zero();
        for &y in &*ys {
            y_sum = y_sum + y;
        }
        drop(ys);

        builder.assert_eq(x, y_sum);
    }
}

type Val = BabyBear;

fn prove_and_verify_sum_air(x: Vec<Val>, ys: Vec<Vec<Val>>) -> Result<(), VerificationError> {
    assert_eq!(x.len(), ys.len());
    let degree = x.len();
    let log_degree = log2_ceil_usize(degree);

    let engine = default_engine(log_degree);

    let x_trace = RowMajorMatrix::new(x, 1);
    let y_width = ys[0].len();
    let y_trace = RowMajorMatrix::new(ys.into_iter().flatten().collect_vec(), y_width);

    let air = SumAir(y_width);

    let mut keygen_builder = engine.keygen_builder();
    let y_ptr = keygen_builder.add_cached_main_matrix(y_width);
    let x_ptr = keygen_builder.add_main_matrix(1);
    keygen_builder.add_partitioned_air(&air, 0, vec![x_ptr, y_ptr]);
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    // Must add trace matrices in the same order as above
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    // Demonstrate y is cached
    let y_data = trace_builder.committer.commit(vec![y_trace.clone()]);
    trace_builder.load_cached_trace(y_trace, y_data);
    // Load x normally
    trace_builder.load_trace(x_trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);
    let pvs = vec![vec![]];

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pvs);

    let vparams = VerificationParams {
        vk,
        proof,
        fri_params: engine.fri_params,
    };
    let (program, input_stream) = common::build_verification_program(pvs, vparams);
    execute_program(program, input_stream);

    Ok(())
}

#[test]
fn test_partitioned_sum_air_happy_path() {
    let rng = StdRng::seed_from_u64(0);
    let n = 1 << 3;
    let ys = generate_random_matrix::<Val>(rng, n, 5);
    let x: Vec<Val> = ys
        .iter()
        .map(|row| row.iter().fold(Val::zero(), |sum, x| sum + *x))
        .collect();
    prove_and_verify_sum_air(x, ys).expect("Verification failed");
}
