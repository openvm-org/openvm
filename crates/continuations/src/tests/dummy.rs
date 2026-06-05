use std::{borrow::BorrowMut, sync::Arc};

use eyre::Result;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{AirProvingContext, DeviceDataTransporter, ProvingContext},
    AirRef, StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2CpuEngine, DuplexSponge, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

pub(in crate::tests) use crate::circuit::deferral::dummy::EmptyAirWithPvs;
use crate::{circuit::deferral::DeferralCircuitPvs, SC};

pub(in crate::tests) fn generate_dummy_def_proof(
    engine: &BabyBearPoseidon2CpuEngine<DuplexSponge>,
    pk: &MultiStarkProvingKey<SC>,
    input_commit: [F; DIGEST_SIZE],
    output_commit: [F; DIGEST_SIZE],
) -> Proof<SC> {
    let mut pvs = vec![F::ZERO; DeferralCircuitPvs::<u8>::width()];
    let pvs_ref: &mut DeferralCircuitPvs<F> = pvs.as_mut_slice().borrow_mut();
    pvs_ref.input_commit = input_commit;
    pvs_ref.output_commit = output_commit;

    let trace = RowMajorMatrix::new(vec![F::ZERO], 1);
    let ctx = ProvingContext {
        per_trace: vec![(
            0,
            AirProvingContext {
                cached_mains: vec![],
                common_main: trace,
                public_values: pvs,
            },
        )],
    };
    let d_pk = engine.device().transport_pk_to_device(pk);
    engine.prove(&d_pk, ctx).unwrap()
}

pub(in crate::tests) fn generate_single_dummy_def_proof(
) -> Result<(Arc<MultiStarkVerifyingKey<SC>>, Proof<SC>)> {
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(super::app_system_params());
    let (pk, vk) = engine
        .keygen(&[Arc::new(EmptyAirWithPvs(DeferralCircuitPvs::<u8>::width())) as AirRef<SC>]);
    let proof = generate_dummy_def_proof(
        &engine,
        &pk,
        [F::ONE; DIGEST_SIZE],
        [F::from_u8(2); DIGEST_SIZE],
    );
    Ok((Arc::new(vk), proof))
}
