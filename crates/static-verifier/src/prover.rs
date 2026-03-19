use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
        flex_gate::MultiPhaseThreadBreakPoints,
    },
    halo2_proofs::{
        dev::MockProver,
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, verify_proof, ProvingKey, VerifyingKey},
        poly::{
            commitment::ParamsProver,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    },
};
use itertools::Itertools;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof},
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use serde::{Deserialize, Serialize};

use crate::{
    config::StaticVerifierShape,
    stages::full_pipeline::{constrained_verify, load_proof_wire},
};

/// KZG parameters for the Halo2 BN256 proving system.
pub type Halo2Params = ParamsKZG<Bn256>;

/// Serializable metadata that accompanies a Halo2 proving key.
///
/// Stores everything needed to reconstruct a prover builder (config params and
/// break points) without the heavyweight [`ProvingKey`] itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halo2ProvingMetadata {
    pub config_params: BaseCircuitParams,
    pub break_points: MultiPhaseThreadBreakPoints,
    pub num_pvs: Vec<usize>,
}

/// A proving key together with the metadata needed to reconstruct prover
/// circuits.
///
/// The [`ProvingKey`] does not implement `Serialize`/`Deserialize` generically
/// for [`BaseCircuitBuilder`]-based circuits (because deserialization requires
/// `circuit-params` support that is not enabled). Use [`Self::pk_to_bytes`] and
/// reconstruct via [`StaticVerifierCircuit::keygen`] when persistence is needed.
#[derive(Debug, Clone)]
pub struct Halo2ProvingPinning {
    pub pk: ProvingKey<G1Affine>,
    pub metadata: Halo2ProvingMetadata,
}

/// Output of [`StaticVerifierCircuit::prove`].
pub struct StaticVerifierProof {
    pub proof_bytes: Vec<u8>,
    pub public_inputs: Vec<Fr>,
}

/// Bundles the three inputs that always travel together when interacting with
/// the static verifier circuit.
pub struct StaticVerifierInput<'a> {
    pub mvk: &'a MultiStarkVerifyingKey<RootConfig>,
    pub proof: &'a Proof<RootConfig>,
}

/// Stateless helper for the static verifier Halo2 circuit.
///
/// Provides circuit-building utilities (`builder`, `populate`), mock proving,
/// real proving, and verification. Keygen is in [`crate::keygen`].
pub struct StaticVerifierCircuit;

impl StaticVerifierCircuit {
    /// Create a [`BaseCircuitBuilder`] configured for the given `stage` and
    /// `shape`.
    pub fn builder(
        stage: CircuitBuilderStage,
        shape: &StaticVerifierShape,
    ) -> BaseCircuitBuilder<Fr> {
        BaseCircuitBuilder::from_stage(stage)
            .use_k(shape.k)
            .use_lookup_bits(shape.lookup_bits)
            .use_instance_columns(shape.instance_columns)
    }

    /// Populate a builder with the static verifier constraints and return the
    /// public inputs.
    pub fn populate(
        builder: &mut BaseCircuitBuilder<Fr>,
        input: &StaticVerifierInput<'_>,
    ) -> Vec<Fr> {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let proof_wire = load_proof_wire(ctx, &range, input.proof);
        let statement_public_inputs =
            constrained_verify(ctx, &range, input.mvk, input.proof, proof_wire);
        let pis = statement_public_inputs
            .iter()
            .map(|c| *c.value())
            .collect_vec();
        builder.assigned_instances[0].extend(statement_public_inputs);

        pis
    }

    /// Run the [`MockProver`] and panic if any constraint is unsatisfied.
    pub fn mock(shape: &StaticVerifierShape, input: &StaticVerifierInput<'_>) {
        let mut builder = Self::builder(CircuitBuilderStage::Mock, shape);
        let public_inputs = Self::populate(&mut builder, input);

        let _ = builder.calculate_params(Some(shape.minimum_rows));

        let prover = MockProver::run(shape.k as u32, &builder, vec![public_inputs])
            .expect("MockProver should initialize");
        prover.assert_satisfied();
    }

    /// Generate a Halo2 proof using a previously computed [`Halo2ProvingPinning`].
    pub fn prove(
        params: &Halo2Params,
        pinning: &Halo2ProvingPinning,
        shape: &StaticVerifierShape,
        input: &StaticVerifierInput<'_>,
    ) -> StaticVerifierProof {
        let mut builder = BaseCircuitBuilder::prover(
            pinning.metadata.config_params.clone(),
            pinning.metadata.break_points.clone(),
        );
        builder = builder.use_instance_columns(shape.instance_columns);

        let public_inputs = Self::populate(&mut builder, input);

        let rng = ChaCha20Rng::from_seed(Default::default());
        let instances: &[&[Fr]] = &[&public_inputs];
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            Challenge255<_>,
            _,
            Blake2bWrite<Vec<u8>, G1Affine, _>,
            _,
        >(
            params,
            &pinning.pk,
            &[builder],
            &[instances],
            rng,
            &mut transcript,
        )
        .expect("Halo2 proof generation should succeed");

        StaticVerifierProof {
            proof_bytes: transcript.finalize(),
            public_inputs,
        }
    }

    /// Verify a Halo2 proof against a verifying key.
    pub fn verify(
        params: &Halo2Params,
        vk: &VerifyingKey<G1Affine>,
        proof: &StaticVerifierProof,
    ) -> bool {
        let verifier_params = params.verifier_params();
        let strategy = SingleStrategy::new(params);
        let instances: &[&[Fr]] = &[&proof.public_inputs];
        let mut transcript =
            Blake2bRead::<_, _, Challenge255<_>>::init(proof.proof_bytes.as_slice());
        verify_proof::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<'_, Bn256>,
            Challenge255<G1Affine>,
            Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
            SingleStrategy<'_, Bn256>,
        >(verifier_params, vk, strategy, &[instances], &mut transcript)
        .is_ok()
    }
}

impl Halo2ProvingPinning {
    /// Serialize the proving key to raw bytes.
    ///
    /// Use this together with [`Halo2ProvingMetadata`] (which is
    /// `Serialize`/`Deserialize`) for persistence.  Re-create the pinning via
    /// [`StaticVerifierCircuit::keygen`] when loading.
    pub fn pk_to_bytes(&self) -> Vec<u8> {
        use halo2_base::halo2_proofs::SerdeFormat;
        self.pk.to_bytes(SerdeFormat::RawBytes)
    }
}
