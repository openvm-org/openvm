use std::sync::Arc;

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
        SerdeFormat,
    },
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::proof::Proof,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    field::baby_bear::{BabyBearChip, BabyBearExtChip},
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
/// ATTN: Deserializer of this struct is not generic. It only works for verifier/wrappr circuit.
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

    /// Run the [`MockProver`] and panic if any constraint is unsatisfied.
    ///
    /// Uses full [`Self::populate`]. Continuations-shaped end-to-end coverage is in `openvm-sdk`
    /// integration tests.
    pub fn mock(&self, shape: &StaticVerifierShape, proof: &Proof<RootConfig>) {
        let mut builder = Self::builder(CircuitBuilderStage::Mock, shape);
        let public_inputs = self.populate(&mut builder, proof);

        let _ = builder.calculate_params(Some(shape.minimum_rows));

        let prover = MockProver::run(shape.k as u32, &builder, vec![public_inputs.to_vec()])
            .expect("MockProver should initialize");
        prover.assert_satisfied();
    }

    /// Prove using only [`Self::populate_verify_stark_constraints`]. `shape.instance_columns` must
    /// be `0`. [`StaticVerifierProof::public_inputs`] is empty.
    pub fn prove_verify_stark_constraints_only(
        &self,
        params: &Halo2Params,
        pinning: &Halo2ProvingPinning,
        shape: &StaticVerifierShape,
        proof: &Proof<RootConfig>,
    ) -> StaticVerifierProof {
        assert_eq!(
            shape.instance_columns, 0,
            "prove_verify_stark_constraints_only requires instance_columns == 0"
        );
        let mut builder = BaseCircuitBuilder::prover(
            pinning.metadata.config_params.clone(),
            pinning.metadata.break_points.clone(),
        );
        builder = builder.use_instance_columns(0);

        let range = builder.range_chip();
        let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
        let ctx = builder.main(0);
        let _ = self.populate_verify_stark_constraints(ctx, &ext_chip, proof);

        let public_inputs = Vec::new();

        let rng = ChaCha20Rng::from_seed(Default::default());
        let instances: &[&[Fr]] = &[];
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

    /// Generate a Halo2 proof using a previously computed [`Halo2ProvingPinning`].
    ///
    /// Uses full [`Self::populate`]. Continuations-shaped proving is covered in `openvm-sdk`
    /// integration tests.
    pub fn prove(
        &self,
        params: &Halo2Params,
        pinning: &Halo2ProvingPinning,
        shape: &StaticVerifierShape,
        proof: &Proof<RootConfig>,
    ) -> StaticVerifierProof {
        let mut builder = BaseCircuitBuilder::prover(
            pinning.metadata.config_params.clone(),
            pinning.metadata.break_points.clone(),
        );
        builder = builder.use_instance_columns(shape.instance_columns);

        let public_inputs = self.populate(&mut builder, proof);
        let public_inputs = public_inputs.to_vec();

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
        let mut transcript =
            Blake2bRead::<_, _, Challenge255<_>>::init(proof.proof_bytes.as_slice());
        // One entry per circuit; inner slice length must match `num_instance_columns` (here: 0).
        let no_instance_cols: &[&[Fr]] = &[];
        let ok = if proof.public_inputs.is_empty() {
            verify_proof::<
                KZGCommitmentScheme<Bn256>,
                VerifierSHPLONK<'_, Bn256>,
                Challenge255<G1Affine>,
                Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
                SingleStrategy<'_, Bn256>,
            >(
                verifier_params,
                vk,
                strategy,
                &[no_instance_cols],
                &mut transcript,
            )
        } else {
            let instances: &[&[Fr]] = &[&proof.public_inputs];
            verify_proof::<
                KZGCommitmentScheme<Bn256>,
                VerifierSHPLONK<'_, Bn256>,
                Challenge255<G1Affine>,
                Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
                SingleStrategy<'_, Bn256>,
            >(verifier_params, vk, strategy, &[instances], &mut transcript)
        };
        ok.is_ok()
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

#[derive(Serialize, Deserialize)]
struct SerializedHalo2ProvingPinning {
    pk_bytes: Vec<u8>,
    metadata: Halo2ProvingMetadata,
}

impl Serialize for Halo2ProvingPinning {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serialized = SerializedHalo2ProvingPinning {
            pk_bytes: self.pk.to_bytes(SerdeFormat::RawBytes),
            metadata: self.metadata.clone(),
        };
        serialized.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Halo2ProvingPinning {
    fn deserialize<D>(deserializer: D) -> Result<Halo2ProvingPinning, D::Error>
    where
        D: Deserializer<'de>,
    {
        let SerializedHalo2ProvingPinning { pk_bytes, metadata } =
            SerializedHalo2ProvingPinning::deserialize(deserializer)?;

        let pk = ProvingKey::<G1Affine>::from_bytes::<BaseCircuitBuilder<Fr>>(
            &pk_bytes,
            SerdeFormat::RawBytes,
            metadata.config_params.clone(),
        )
        .map_err(|e| de::Error::custom(format!("invalid bytes for proving key: {e}")))?;

        Ok(Halo2ProvingPinning { pk, metadata })
    }
}
