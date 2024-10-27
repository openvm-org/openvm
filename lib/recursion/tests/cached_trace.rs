// use afs_compiler::util::execute_program;
// use axvm_recursion::testing_utils::inner::build_verification_program;
// use ax_stark_backend::{
//     air_builders::PartitionedAirBuilder,
//     engine::VerificationData,
//     prover::trace::TraceCommitmentBuilder,
//     rap::{BaseAirWithPublicValues, PartitionedBaseAir},
//     verifier::VerificationError,
// };
// use ax_sdk::{
//     config::baby_bear_poseidon2::default_engine,
//     engine::{StarkEngine, VerificationDataWithFriParams},
//     utils::generate_random_matrix,
// };
// use itertools::Itertools;
// use p3_air::{Air, BaseAir};
// use p3_baby_bear::BabyBear;
// use p3_field::AbstractField;
// use p3_matrix::{dense::RowMajorMatrix, Matrix};
// use p3_util::log2_ceil_usize;
// use rand::{rngs::StdRng, SeedableRng};
//
// /// Inner value is width of y-submatrix
// pub struct SumAir(pub usize);
//
// impl<F> BaseAirWithPublicValues<F> for SumAir {}
// impl<F> PartitionedBaseAir<F> for SumAir {
//     fn cached_main_widths(&self) -> Vec<usize> {
//         vec![self.0]
//     }
//     fn common_main_width(&self) -> usize {
//         1
//     }
// }
// impl<F> BaseAir<F> for SumAir {
//     fn width(&self) -> usize {
//         self.0 + 1
//     }
// }
//
// impl<AB: PartitionedAirBuilder> Air<AB> for SumAir {
//     fn eval(&self, builder: &mut AB) {
//         assert_eq!(builder.cached_mains().len(), 1);
//
//         let x = builder.common_main().row_slice(0)[0];
//         let ys = builder.cached_mains()[0].row_slice(0);
//
//         let mut y_sum = AB::Expr::zero();
//         for &y in &*ys {
//             y_sum = y_sum + y;
//         }
//         drop(ys);
//
//         builder.assert_eq(x, y_sum);
//     }
// }
//
// type Val = BabyBear;
//
// fn prove_and_verify_sum_air(x: Vec<Val>, ys: Vec<Vec<Val>>) -> Result<(), VerificationError> {
//     assert_eq!(x.len(), ys.len());
//     let degree = x.len();
//     let log_degree = log2_ceil_usize(degree);
//
//     let engine = default_engine(log_degree);
//
//     let x_trace = RowMajorMatrix::new(x, 1);
//     let y_width = ys[0].len();
//     let y_trace = RowMajorMatrix::new(ys.into_iter().flatten().collect_vec(), y_width);
//
//     let air = SumAir(y_width);
//
//     let mut keygen_builder = engine.keygen_builder();
//     let y_ptr = keygen_builder.add_cached_main_matrix(y_width);
//     let x_ptr = keygen_builder.add_main_matrix(1);
//     keygen_builder.add_partitioned_air(&air, vec![y_ptr, x_ptr]);
//     let pk = keygen_builder.generate_pk();
//     let vk = pk.vk();
//
//     let prover = engine.prover();
//     // Must add trace matrices in the same order as above
//     let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
//     // Demonstrate y is cached
//     let y_data = trace_builder.committer.commit(vec![y_trace.clone()]);
//     trace_builder.load_cached_trace(y_trace, y_data);
//     // Load x normally
//     trace_builder.load_trace(x_trace);
//     trace_builder.commit_current();
//
//     let main_trace_data = trace_builder.view(&vk, vec![&air]);
//     let pvs = vec![vec![]];
//
//     let mut challenger = engine.new_challenger();
//     let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pvs);
//
//     let vparams = VerificationDataWithFriParams {
//         data: VerificationData { vk, proof },
//         fri_params: engine.fri_params,
//     };
//     let (program, input_stream) = build_verification_program(vparams, Default::default());
//     execute_program(program, input_stream);
//
//     Ok(())
// }
//
// #[test]
// fn test_partitioned_sum_air_happy_path() {
//     let rng = StdRng::seed_from_u64(0);
//     let n = 1 << 3;
//     let ys = generate_random_matrix::<Val>(rng, n, 5);
//     let x: Vec<Val> = ys
//         .iter()
//         .map(|row| row.iter().fold(Val::zero(), |sum, x| sum + *x))
//         .collect();
//     prove_and_verify_sum_air(x, ys).expect("Verification failed");
// }
