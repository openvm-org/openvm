use std::{borrow::BorrowMut, sync::Arc};

use openvm::platform::memory::MEM_SIZE;
use openvm_sdk::{
    config::{default_system_params, AggregationSystemParams},
    Sdk, StdIn,
};
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
    test_utils::{PreprocessedAndCachedFixture, TestFixture},
    AirRef, PartitionedBaseAir, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, BabyBearPoseidon2RefEngine, DuplexSponge, F,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_host::{
    pvs::VerifierBasePvs, verify_vm_stark_proof_pvs, vk::VmStarkVerifyingKey, VmStarkProof,
};

pub(super) const OPTIONAL_AIR_ARITY: usize = 3;
pub(super) const OPTIONAL_AIR_ID: usize = 4;
const OPTIONAL_TRACE_HEIGHT: usize = 32;

pub(super) struct Fixture {
    pub(super) vk: VmStarkVerifyingKey,
    pub(super) proof: VmStarkProof,
}

struct PublicValuesAir {
    arity: usize,
    trace_width: usize,
}

impl PartitionedBaseAir<F> for PublicValuesAir {}

impl BaseAir<F> for PublicValuesAir {
    fn width(&self) -> usize {
        self.trace_width
    }
}

impl BaseAirWithPublicValues<F> for PublicValuesAir {
    fn num_public_values(&self) -> usize {
        self.arity
    }
}

impl<AB: AirBuilderWithPublicValues<F = F>> Air<AB> for PublicValuesAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("main trace has a local row");
        let public_values = builder.public_values().to_vec();
        let mut first_row = builder.when_first_row();
        for (cell, public_value) in local.iter().zip(public_values) {
            first_row.assert_eq(cell.clone(), public_value);
        }
        for cell in local.iter().skip(self.arity) {
            builder.assert_zero(cell.clone());
        }
    }
}

struct OptionalAirFixture {
    config: SC,
    public_values: [Vec<F>; 4],
    optional_air_present: bool,
}

impl OptionalAirFixture {
    fn cached_fixture(&self) -> PreprocessedAndCachedFixture<SC> {
        PreprocessedAndCachedFixture::new(
            (0..OPTIONAL_TRACE_HEIGHT).map(|idx| idx % 2 == 0).collect(),
            self.config.clone(),
            1,
        )
    }
}

impl TestFixture<SC> for OptionalAirFixture {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let public_values_air = |values: &Vec<F>| {
            Arc::new(PublicValuesAir {
                arity: values.len(),
                trace_width: values.len().max(1),
            }) as AirRef<SC>
        };
        self.public_values[..3]
            .iter()
            .map(public_values_air)
            .chain(self.cached_fixture().airs())
            .chain(self.public_values[3..].iter().map(|values| {
                Arc::new(PublicValuesAir {
                    arity: values.len(),
                    trace_width: values.len(),
                }) as AirRef<SC>
            }))
            .collect()
    }

    fn generate_proving_ctx(&self) -> ProvingContext<CpuColMajorBackend<SC>> {
        let mut per_trace: Vec<_> = self
            .public_values
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != 3 || self.optional_air_present)
            .map(|(idx, public_values)| {
                let air_idx = if idx == 3 { OPTIONAL_AIR_ID } else { idx };
                let trace_width = public_values.len().max(1);
                let rows = if public_values.is_empty() {
                    vec![F::ZERO; OPTIONAL_TRACE_HEIGHT]
                } else {
                    (0..OPTIONAL_TRACE_HEIGHT)
                        .flat_map(|_| public_values.iter().copied())
                        .collect()
                };
                let trace = RowMajorMatrix::new(rows, trace_width);
                (
                    air_idx,
                    AirProvingContext::simple(
                        ColMajorMatrix::from_row_major(&trace),
                        public_values.clone(),
                    ),
                )
            })
            .collect();
        let cached_ctx = self
            .cached_fixture()
            .generate_proving_ctx()
            .per_trace
            .into_iter()
            .next()
            .expect("cached fixture has one AIR")
            .1;
        per_trace.push((3, cached_ctx));
        ProvingContext::new(per_trace)
    }
}

pub(super) fn fixture() -> Fixture {
    let sdk = Sdk::riscv32(default_system_params(), AggregationSystemParams::default());
    let elf = Elf::decode(
        include_bytes!("../../../sdk/programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )
    .expect("decode Fibonacci ELF");
    let exe = sdk
        .convert_to_exe(elf)
        .expect("convert Fibonacci executable");
    let mut stdin = StdIn::default();
    stdin.write(&100u64);
    let (proof, baseline) = sdk.prove(exe, stdin, &[]).expect("prove Fibonacci");
    let vk = VmStarkVerifyingKey {
        mvk: (*sdk.agg_vk()).clone(),
        baseline,
    };
    Sdk::verify_proof(vk.mvk.clone(), vk.baseline.clone(), &proof)
        .expect("Rust VM verifier accepts generated proof");
    Fixture { vk, proof }
}

pub(super) fn optional_air_proofs() -> (VmStarkVerifyingKey, VmStarkProof, VmStarkProof) {
    let canonical = fixture();
    let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(SystemParams::new_for_testing(5));
    let mut public_values = [
        canonical.proof.inner.public_values[0].clone(),
        canonical.proof.inner.public_values[1].clone(),
        Vec::new(),
        vec![F::ONE; OPTIONAL_AIR_ARITY],
    ];

    let calibration_fixture = OptionalAirFixture {
        config: engine.config().clone(),
        public_values: public_values.clone(),
        optional_air_present: true,
    };
    let (pk, mvk) = calibration_fixture.keygen(&engine);
    let calibration_proof = calibration_fixture.prove(&engine, &pk);
    let cached_commit = calibration_proof.trace_vdata[3]
        .as_ref()
        .expect("cached AIR is present")
        .cached_commitments[0];

    let mut baseline = canonical.vk.baseline;
    let verifier_pvs: &mut VerifierBasePvs<F> = public_values[0].as_mut_slice().borrow_mut();
    if verifier_pvs.recursion_depth == F::ONE {
        verifier_pvs.internal_for_leaf_vk_commit.cached_commit = cached_commit;
        baseline.internal_for_leaf_vk_commit.cached_commit = cached_commit;
    } else {
        verifier_pvs.internal_recursive_vk_commit.cached_commit = cached_commit;
        baseline.internal_recursive_vk_commit.cached_commit = cached_commit;
    }

    let present_fixture = OptionalAirFixture {
        config: engine.config().clone(),
        public_values: public_values.clone(),
        optional_air_present: true,
    };
    let absent_fixture = OptionalAirFixture {
        config: engine.config().clone(),
        public_values,
        optional_air_present: false,
    };
    let present_inner = present_fixture.prove(&engine, &pk);
    let absent_inner = absent_fixture.prove(&engine, &pk);
    engine
        .verify(&mvk, &present_inner)
        .expect("Rust accepts the present optional AIR");
    engine
        .verify(&mvk, &absent_inner)
        .expect("Rust accepts the absent optional AIR");

    let user_pvs_proof = canonical.proof.user_pvs_proof;
    let wrap = |inner| VmStarkProof {
        inner,
        user_pvs_proof: user_pvs_proof.clone(),
        deferral_merkle_proofs: None,
    };
    let vk = VmStarkVerifyingKey { mvk, baseline };
    let present_proof = wrap(present_inner);
    let absent_proof = wrap(absent_inner);
    verify_vm_stark_proof_pvs(&vk, &present_proof)
        .expect("Rust VM claims accept the present optional AIR");
    verify_vm_stark_proof_pvs(&vk, &absent_proof)
        .expect("Rust VM claims accept the absent optional AIR");
    (vk, present_proof, absent_proof)
}
