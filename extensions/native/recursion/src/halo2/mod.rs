pub mod utils;
pub mod verifier;

pub mod testing_utils;
#[cfg(test)]
mod tests;
pub mod wrapper;

use std::fmt::Debug;

use itertools::Itertools;
use openvm_native_compiler::{
    constraints::halo2::compiler::{Halo2ConstraintCompiler, Halo2State},
    ir::{Config, DslIr, TracedVec, Witness},
};
use openvm_stark_backend::p3_field::extension::BinomialExtensionField;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, p3_bn254_fr::Bn254Fr};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use snark_verifier_sdk::{
    halo2::{gen_dummy_snark_from_vk, gen_snark_shplonk},
    snark_verifier::halo2_base::{
        gates::{
            circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
            flex_gate::MultiPhaseThreadBreakPoints,
        },
        halo2_proofs::{
            dev::MockProver,
            halo2curves::bn256::{Bn256, G1Affine},
            plonk::{keygen_pk2, ProvingKey},
            poly::{commitment::Params, kzg::commitment::ParamsKZG},
            SerdeFormat,
        },
    },
    CircuitExt, Snark, SHPLONK,
};

use crate::halo2::utils::Halo2ParamsReader;

pub type Halo2Params = ParamsKZG<Bn256>;
pub use snark_verifier_sdk::snark_verifier::halo2_base::halo2_proofs::halo2curves::bn256::Fr;

/// A prover that can generate proofs with the Halo2
#[derive(Debug, Clone)]
pub struct Halo2Prover;

#[derive(Clone, Deserialize, Serialize)]
pub struct RawEvmProof {
    pub instances: Vec<Fr>,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DslOperations<C: Config> {
    pub operations: TracedVec<DslIr<C>>,
    pub num_public_values: usize,
}

/// Necessary metadata to prove a Halo2 circuit
/// Attention: Deserializer of this struct is not generic. It only works for verifier/wrapper
/// circuit.
#[derive(Debug, Clone)]
pub struct Halo2ProvingPinning {
    pub pk: ProvingKey<G1Affine>,
    pub metadata: Halo2ProvingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halo2ProvingMetadata {
    pub config_params: BaseCircuitParams,
    pub break_points: MultiPhaseThreadBreakPoints,
    /// Number of public values per column in order.
    pub num_pvs: Vec<usize>,
}

impl RawEvmProof {
    /// Return bytes calldata to be passed to the verifier contract.
    #[cfg(feature = "evm-prove")]
    pub fn verifier_calldata(&self) -> Vec<u8> {
        snark_verifier_sdk::evm::encode_calldata(&[self.instances.clone()], &self.proof)
    }
}

impl Halo2ProvingPinning {
    pub fn generate_dummy_snark(&self, reader: &impl Halo2ParamsReader) -> Snark {
        let k = self.metadata.config_params.k;
        let params = reader.read_params(k);
        gen_dummy_snark_from_vk::<SHPLONK>(
            &params,
            self.pk.get_vk(),
            self.metadata.num_pvs.clone(),
            None,
        )
    }
}

impl Halo2Prover {
    pub fn builder(stage: CircuitBuilderStage, k: usize) -> BaseCircuitBuilder<Fr> {
        BaseCircuitBuilder::from_stage(stage)
            .use_k(k)
            .use_lookup_bits(k - 1)
            .use_instance_columns(1)
    }

    pub fn populate<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        builder: BaseCircuitBuilder<Fr>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
        #[allow(unused_variables)] profiling: bool,
    ) -> BaseCircuitBuilder<Fr> {
        let mut state = Halo2State {
            builder,
            ..Default::default()
        };
        state.load_witness(witness);

        let backend = Halo2ConstraintCompiler::<C>::new(dsl_operations.num_public_values);
        #[cfg(feature = "metrics")]
        let backend = if profiling {
            backend.with_profiling()
        } else {
            backend
        };
        backend.constrain_halo2(&mut state, dsl_operations.operations);

        state.builder
    }

    /// Executes the prover in testing mode with a circuit definition and witness.
    ///
    /// Returns the public instances.
    pub fn mock<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Vec<Vec<Fr>> {
        let builder = Self::builder(CircuitBuilderStage::Mock, k);
        let mut builder = Self::populate(builder, dsl_operations, witness, true);

        let public_instances = builder.instances();
        println!("Public instances: {:?}", public_instances);

        builder.calculate_params(Some(20));

        MockProver::run(k as u32, &builder, public_instances.clone())
            .unwrap()
            .assert_satisfied();
        public_instances
    }

    /// Populates builder, tunes circuit, keygen
    pub fn keygen<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        params: &Halo2Params,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Halo2ProvingPinning {
        let k = params.k() as usize;
        let builder = Self::builder(CircuitBuilderStage::Keygen, k);
        let mut builder = Self::populate(builder, dsl_operations, witness, true);
        builder.calculate_params(Some(20));

        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let pk = keygen_pk2(params, &builder, false).unwrap();
        #[cfg(feature = "metrics")]
        metrics::gauge!("halo2_keygen_time_ms").set(start.elapsed().as_millis() as f64);
        let break_points = builder.break_points();

        let config_params = builder.config_params.clone();
        let num_pvs = builder
            .assigned_instances
            .iter()
            .map(|x| x.len())
            .collect_vec();

        Halo2ProvingPinning {
            pk,
            metadata: Halo2ProvingMetadata {
                config_params,
                break_points,
                num_pvs,
            },
        }
    }

    pub fn prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        params: &Halo2Params,
        config_params: BaseCircuitParams,
        break_points: MultiPhaseThreadBreakPoints,
        pk: &ProvingKey<G1Affine>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
        profiling: bool,
    ) -> Snark {
        let k = config_params.k;
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let builder = Self::builder(CircuitBuilderStage::Prover, k)
            .use_params(config_params)
            .use_break_points(break_points);
        let builder = Self::populate(builder, dsl_operations, witness, profiling);
        #[cfg(feature = "metrics")]
        {
            let stats = builder.statistics();
            let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
            let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
            let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
            metrics::counter!("main_cells_used").absolute(total_cell as u64);
        }
        let snark = gen_snark_shplonk(params, pk, builder, None::<&str>);

        #[cfg(feature = "metrics")]
        metrics::gauge!("total_proof_time_ms").set(start.elapsed().as_millis() as f64);

        snark
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
        .map_err(|e| de::Error::custom(format!("invalid bytes for proving key: {}", e)))?;

        Ok(Halo2ProvingPinning { pk, metadata })
    }
}
