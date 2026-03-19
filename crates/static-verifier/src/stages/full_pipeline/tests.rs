use halo2_base::{
    gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as RootConfig, BabyBearBn254Poseidon2CpuEngine,
    },
    openvm_stark_backend::{
        p3_field::{PrimeCharacteristicRing, PrimeField64, TwoAdicField},
        test_utils::{
            test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
            PreprocessedFibFixture, TestFixture,
        },
        StarkEngine,
    },
};

use super::*;
use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS, STATIC_VERIFIER_NUM_ADVICE_COLS},
    field::baby_bear::{
        clear_recorded_ext_base_consts, take_recorded_ext_base_consts, RecordedExtBaseConst,
        BABY_BEAR_MODULUS_U64,
    },
    prover::{StaticVerifierCircuit, StaticVerifierInput},
    RootF,
};

const END_TO_END_K: u32 = 22;
const END_TO_END_LOOKUP_BITS: usize = 8;
const END_TO_END_MIN_ROWS: usize = 32768;

fn run_mock(
    expect_satisfied: bool,
    public_inputs: &[Fr],
    build: impl FnOnce(&mut BaseCircuitBuilder<Fr>),
) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(END_TO_END_K as usize)
        .use_lookup_bits(END_TO_END_LOOKUP_BITS)
        .use_instance_columns(1);

    if expect_satisfied {
        build(&mut builder);
    } else {
        // Disable guarded debug assertions in BabyBearChip, and catch host-side
        // panics (e.g. deterministic metadata shape checks) that fire before the
        // MockProver can verify constraints.
        let build_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::utils::with_debug_asserts_disabled(|| build(&mut builder));
        }));
        if build_result.is_err() {
            return;
        }
    }

    let params = builder.calculate_params(Some(END_TO_END_MIN_ROWS));
    assert!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_NUM_ADVICE_COLS
    );
    assert!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_LOOKUP_ADVICE_COLS
    );

    let prover = MockProver::run(END_TO_END_K, &builder, vec![public_inputs.to_vec()])
        .expect("mock prover should initialize for pipeline end-to-end circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected pipeline end-to-end constraints to fail",
        );
    }
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

fn pipeline_public_inputs(
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    proof: &Proof<RootConfig>,
) -> Vec<Fr> {
    vec![
        digest_scalar_to_fr(mvk.pre_hash[0]),
        digest_scalar_to_fr(proof.common_main_commit[0]),
    ]
}

fn assert_fixture_matches_native<Fx>(engine: &BabyBearBn254Poseidon2CpuEngine, fixture: Fx)
where
    Fx: TestFixture<RootConfig>,
{
    let (vk, proof) = fixture.keygen_and_prove(engine);
    let public_inputs = pipeline_public_inputs(&vk, &proof);

    run_mock(true, &public_inputs, |builder| {
        StaticVerifierCircuit::populate(
            builder,
            &StaticVerifierInput {
                mvk: &vk,
                proof: &proof,
            },
        );
    });
}

fn prank_recorded_ext_constant(
    ctx: &mut Context<Fr>,
    records: &[RecordedExtBaseConst],
    family: &str,
    constant: u64,
) {
    let record = records
        .iter()
        .find(|record| record.constant == constant)
        .unwrap_or_else(|| panic!("missing recorded ext-base constant for {family}={constant}"));
    record
        .cell
        .debug_prank(ctx, Fr::from((constant + 1) % BABY_BEAR_MODULUS_U64));
}

#[test]
fn pipeline_end_to_end_matches_native_for_fib_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, FibFixture::new(0, 1, 1 << 5));
}

#[test]
fn pipeline_end_to_end_matches_native_for_interactions_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, InteractionsFixture11);
}

#[test]
fn pipeline_end_to_end_matches_native_for_cached_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, CachedFixture11::new(engine.config().clone()));
}

#[test]
fn pipeline_end_to_end_matches_native_for_preprocessed_fixture() {
    let engine = test_engine();
    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect::<Vec<_>>();
    assert_fixture_matches_native(&engine, PreprocessedFibFixture::new(0, 1, sels));
}

#[test]
fn pipeline_constraints_fail_when_ext_constant_families_are_pranked() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    let l_skip = vk.inner.params.l_skip;
    let subgroup_root = RootF::two_adic_generator(l_skip).as_canonical_u64();
    let bus_constant = vk
        .inner
        .per_air
        .iter()
        .flat_map(|air| air.symbolic_constraints.interactions.iter())
        .map(|interaction| u64::from(interaction.bus_index) + 1)
        .find(|&value| value > 1)
        .unwrap_or(1);
    let normalization_family_constants = (1..=31usize)
        .map(|pow| {
            (0..pow)
                .fold(RootF::ONE, |acc, _| acc.halve())
                .as_canonical_u64()
        })
        .collect::<Vec<_>>();
    let base_families = [
        ("one", 1u64),
        ("two", 2u64),
        ("subgroup_root", subgroup_root),
        ("bus_index_plus_one", bus_constant),
    ];

    let public_inputs = pipeline_public_inputs(&vk, &proof);
    run_mock(false, &public_inputs, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let statement_public_inputs = [
            ctx.load_witness(digest_scalar_to_fr(vk.pre_hash[0])),
            ctx.load_witness(digest_scalar_to_fr(proof.common_main_commit[0])),
        ];
        clear_recorded_ext_base_consts();
        constrained_verify(ctx, &range, &vk, &proof, statement_public_inputs);
        let records = take_recorded_ext_base_consts();
        for (family, constant) in base_families {
            prank_recorded_ext_constant(ctx, &records, family, constant);
        }
        let normalization_constant = records
            .iter()
            .find(|record| normalization_family_constants.contains(&record.constant))
            .map(|record| record.constant)
            .unwrap_or(1);
        prank_recorded_ext_constant(ctx, &records, "normalization", normalization_constant);
        builder.assigned_instances[0].extend(statement_public_inputs);
    });
}

#[test]
fn pipeline_constraints_fail_when_public_inputs_are_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut tampered_public_inputs = pipeline_public_inputs(&vk, &proof);
    tampered_public_inputs[0] += Fr::from(1u64);

    run_mock(false, &tampered_public_inputs, |builder| {
        StaticVerifierCircuit::populate(
            builder,
            &StaticVerifierInput {
                mvk: &vk,
                proof: &proof,
            },
        );
    });
}
