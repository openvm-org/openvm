//! Shared test fixture: a small STARK proof shaped like a root proof, so the
//! production `populate_pvs` path runs without any cached artifacts.

use std::{sync::Arc, time::Instant};

use openvm_continuations::circuit::root::RootVerifierPvs;
use openvm_stark_sdk::{
    config::{
        baby_bear_bn254_poseidon2::{
            BabyBearBn254Poseidon2Config as RootConfig, BabyBearBn254Poseidon2CpuEngine,
        },
        baby_bear_poseidon2::{Digest as InnerDigest, DIGEST_SIZE},
    },
    openvm_stark_backend::{
        p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues},
        p3_field::PrimeCharacteristicRing,
        p3_matrix::{dense::RowMajorMatrix, Matrix},
        proof::Proof,
        prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
        test_utils::{test_system_params_small, InteractionsFixture11, TestFixture},
        AirRef, PartitionedBaseAir, StarkEngine,
    },
};
use openvm_verify_stark_host::pvs::CONSTRAINT_EVAL_AIR_ID;

use crate::{stages::proof_shape::log_heights_per_air_from_proof, RootF, StaticVerifierCircuit};

/// Small keygen k: fast in-memory SRS setup, and the advice tape spans several
/// columns so break-point and lookup round-robin placement are exercised.
pub const FIXTURE_K: usize = 19;

/// Test-only AIR whose `num_pvs` public values are bound to the first row of
/// an equally-wide trace.
struct PvAir {
    num_pvs: usize,
}

impl<F> PartitionedBaseAir<F> for PvAir {}
impl<F> BaseAir<F> for PvAir {
    fn width(&self) -> usize {
        self.num_pvs
    }
}
impl<F> BaseAirWithPublicValues<F> for PvAir {
    fn num_public_values(&self) -> usize {
        self.num_pvs
    }
}
impl<AB: AirBuilderWithPublicValues> Air<AB> for PvAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let pis = builder.public_values().to_vec();
        let mut when_first_row = builder.when_first_row();
        for (cell, pi) in local.iter().zip(pis) {
            when_first_row.assert_eq(cell.clone(), pi);
        }
    }
}

const NUM_USER_PVS: usize = 4;
const PV_AIR_HEIGHT: usize = 4;

/// Fixture whose first four AIRs mirror a root proof's public-value shape —
/// `RootVerifierPvs` at AIR 0, user PVs at AIR 1, `DagCommitPvs` at AIR 3 — so
/// the production `populate_pvs` path (PV extraction + dag onion-commit pin)
/// runs on a fixture proof. An interaction AIR pair keeps logup exercised.
struct RootShapedFixture;

impl RootShapedFixture {
    fn pv_counts() -> [usize; 4] {
        [
            size_of::<RootVerifierPvs<u8>>(),
            NUM_USER_PVS,
            1,
            DIGEST_SIZE, // DagCommitPvs
        ]
    }

    fn pvs(air_idx: usize, num_pvs: usize) -> Vec<RootF> {
        (0..num_pvs)
            .map(|i| RootF::from_usize(1 + 100 * air_idx + i))
            .collect()
    }
}

impl TestFixture<RootConfig> for RootShapedFixture {
    fn airs(&self) -> Vec<AirRef<RootConfig>> {
        Self::pv_counts()
            .into_iter()
            .map(|num_pvs| Arc::new(PvAir { num_pvs }) as AirRef<_>)
            .chain(InteractionsFixture11.airs())
            .collect()
    }

    fn generate_proving_ctx(&self) -> ProvingContext<CpuColMajorBackend<RootConfig>> {
        let pv_counts = Self::pv_counts();
        let mut per_trace: Vec<_> = pv_counts
            .iter()
            .enumerate()
            .map(|(air_idx, &num_pvs)| {
                let pvs = Self::pvs(air_idx, num_pvs);
                let rows: Vec<RootF> = std::iter::repeat_with(|| pvs.clone())
                    .take(PV_AIR_HEIGHT)
                    .flatten()
                    .collect();
                (
                    air_idx,
                    AirProvingContext::simple(
                        ColMajorMatrix::from_row_major(&RowMajorMatrix::new(rows, num_pvs)),
                        pvs,
                    ),
                )
            })
            .collect();
        per_trace.extend(
            InteractionsFixture11
                .generate_proving_ctx()
                .per_trace
                .into_iter()
                .map(|(air_idx, ctx)| (air_idx + pv_counts.len(), ctx)),
        );
        ProvingContext::new(per_trace)
    }
}

/// STARK-proves [`RootShapedFixture`] and builds the matching
/// [`StaticVerifierCircuit`] (the dag onion commit is pinned to the fixture
/// proof's own AIR-3 public values, so `populate_pvs` accepts it).
pub fn fixture_circuit_and_proof() -> (StaticVerifierCircuit, Proof<RootConfig>) {
    let engine: BabyBearBn254Poseidon2CpuEngine =
        BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3));
    let start = Instant::now();
    let (vk, proof) = RootShapedFixture.keygen_and_prove(&engine);
    println!("fixture STARK keygen + prove: {:?}", start.elapsed());
    let log_heights_per_air = log_heights_per_air_from_proof(&proof);
    let onion_commit: InnerDigest = proof.public_values[CONSTRAINT_EVAL_AIR_ID]
        .as_slice()
        .try_into()
        .unwrap();
    let circuit = StaticVerifierCircuit::try_new(vk, onion_commit, &log_heights_per_air)
        .expect("static circuit params");
    (circuit, proof)
}
