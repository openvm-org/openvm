#[cfg(feature = "evm-prove")]
pub mod evm;

use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
        flex_gate::MultiPhaseThreadBreakPoints,
        GateInstructions, RangeInstructions,
    },
    halo2_proofs::{
        dev::MockProver,
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey},
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
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as NativeConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof},
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use serde::{Deserialize, Serialize};

use crate::{
    config::StaticVerifierShape,
    stages::full_pipeline::{derive_and_constrain_pipeline, derive_pipeline_public_inputs},
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
/// reconstruct via [`Halo2Prover::keygen`] when persistence is needed.
#[derive(Debug, Clone)]
pub struct Halo2ProvingPinning {
    pub pk: ProvingKey<G1Affine>,
    pub metadata: Halo2ProvingMetadata,
}

/// Output of [`Halo2Prover::prove`].
pub struct StaticVerifierProof {
    pub proof_bytes: Vec<u8>,
    pub public_inputs: Vec<Fr>,
}

/// Bundles the three inputs that always travel together when interacting with
/// the static verifier circuit.
pub struct StaticVerifierInput<'a> {
    pub config: &'a NativeConfig,
    pub mvk: &'a MultiStarkVerifyingKey<NativeConfig>,
    pub proof: &'a Proof<NativeConfig>,
}

/// Low-level, stateless prover for the static verifier Halo2 circuit.
///
/// All methods are associated functions (no `self`). This mirrors the
/// `Halo2Prover` pattern from `openvm-native-recursion`.
pub struct Halo2Prover;

impl Halo2Prover {
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
    ///
    /// This is the core circuit-building logic extracted from the test helper
    /// `build_end_to_end_constraints_from_proof`.
    pub fn populate(
        builder: &mut BaseCircuitBuilder<Fr>,
        input: &StaticVerifierInput<'_>,
    ) -> Vec<Fr> {
        let range = builder.range_chip();
        let public_input_cells = {
            let ctx = builder.main(0);
            let assigned =
                derive_and_constrain_pipeline(ctx, &range, input.config, input.mvk, input.proof)
                    .expect("pipeline derive+constrain should succeed");

            range
                .gate()
                .assert_is_const(ctx, &assigned.whir.mu_pow_witness_ok, &Fr::from(1u64));

            assigned.statement_public_inputs.to_vec()
        };
        builder.assigned_instances[0].extend(public_input_cells);

        derive_pipeline_public_inputs(input.config, input.mvk, input.proof)
    }

    /// Compute the public inputs without building any circuit constraints.
    pub fn public_inputs(input: &StaticVerifierInput<'_>) -> Vec<Fr> {
        derive_pipeline_public_inputs(input.config, input.mvk, input.proof)
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

    /// Run keygen to produce a [`Halo2ProvingPinning`].
    ///
    /// The `input` is used as a representative witness for keygen; any valid
    /// input for the target circuit shape will do.
    pub fn keygen(
        params: &Halo2Params,
        shape: &StaticVerifierShape,
        input: &StaticVerifierInput<'_>,
    ) -> Halo2ProvingPinning {
        let mut builder = Self::builder(CircuitBuilderStage::Keygen, shape);
        let public_inputs = Self::populate(&mut builder, input);

        let config_params = builder.calculate_params(Some(shape.minimum_rows));

        let vk = keygen_vk(params, &builder).expect("keygen_vk should succeed");
        let pk = keygen_pk(params, vk, &builder).expect("keygen_pk should succeed");
        let break_points = builder.break_points();

        Halo2ProvingPinning {
            pk,
            metadata: Halo2ProvingMetadata {
                config_params,
                break_points,
                num_pvs: vec![public_inputs.len()],
            },
        }
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
        // Re-configure instance columns to match the shape.
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
    /// [`Halo2Prover::keygen`] when loading.
    pub fn pk_to_bytes(&self) -> Vec<u8> {
        use halo2_base::halo2_proofs::SerdeFormat;
        self.pk.to_bytes(SerdeFormat::RawBytes)
    }
}

/// High-level proving key that owns a [`Halo2ProvingPinning`] together with
/// the [`StaticVerifierShape`] needed to reconstruct prover builders.
///
/// This is analogous to `Halo2WrapperProvingKey` from `openvm-native-recursion`.
pub struct StaticVerifierProvingKey {
    pub pinning: Halo2ProvingPinning,
    pub shape: StaticVerifierShape,
}

impl StaticVerifierProvingKey {
    /// Run keygen and return a proving key that can be reused for multiple
    /// proofs.
    pub fn keygen(
        params: &Halo2Params,
        shape: StaticVerifierShape,
        input: &StaticVerifierInput<'_>,
    ) -> Self {
        let pinning = Halo2Prover::keygen(params, &shape, input);
        Self { pinning, shape }
    }

    /// Generate a proof using the stored pinning and shape.
    pub fn prove(
        &self,
        params: &Halo2Params,
        input: &StaticVerifierInput<'_>,
    ) -> StaticVerifierProof {
        Halo2Prover::prove(params, &self.pinning, &self.shape, input)
    }

    /// Verify a proof against this proving key's verifying key.
    pub fn verify(&self, params: &Halo2Params, proof: &StaticVerifierProof) -> bool {
        Halo2Prover::verify(params, self.pinning.pk.get_vk(), proof)
    }
}
