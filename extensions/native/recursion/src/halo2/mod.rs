pub mod utils;
pub mod verifier;

pub mod testing_utils;
#[cfg(test)]
mod tests;
pub mod wrapper;

use std::{fmt, fmt::Debug};

use axvm_native_compiler::{
    constraints::halo2::compiler::{Halo2ConstraintCompiler, Halo2State},
    ir::{Config, DslIr, TracedVec, Witness},
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::extension::BinomialExtensionField;
use serde::{
    de,
    de::{MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use snark_verifier_sdk::{
    halo2::{gen_dummy_snark_from_vk, gen_snark_shplonk},
    snark_verifier::halo2_base::{
        gates::{
            circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
            flex_gate::MultiPhaseThreadBreakPoints,
        },
        halo2_proofs::{
            dev::MockProver,
            halo2curves::bn256::{Bn256, Fr, G1Affine},
            plonk::{keygen_pk2, ProvingKey},
            poly::kzg::commitment::ParamsKZG,
            SerdeFormat,
        },
    },
    CircuitExt, Snark, SHPLONK,
};

use crate::halo2::utils::read_params;

pub type Halo2Params = ParamsKZG<Bn256>;

/// A prover that can generate proofs with the Halo2
#[derive(Debug, Clone)]
pub struct Halo2Prover;

#[derive(Clone, Deserialize, Serialize)]
pub struct EvmProof {
    pub instances: Vec<Vec<Fr>>,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DslOperations<C: Config> {
    pub operations: TracedVec<DslIr<C>>,
    pub num_public_values: usize,
}

/// Necessary metadata to prove a Halo2 circuit
/// Attention: Deserializer of this struct is not generic. It only works for verifier/wrapper circuit.
#[derive(Debug, Clone, Serialize)]
pub struct Halo2ProvingPinning {
    #[serde(serialize_with = "pk_serializer")]
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

impl Halo2ProvingPinning {
    pub fn generate_dummy_snark(&self) -> Snark {
        let k = self.metadata.config_params.k;
        let params = read_params(k as u32);
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
        #[allow(unused_variables)] collect_metrics: bool,
    ) -> BaseCircuitBuilder<Fr> {
        let mut state = Halo2State {
            builder,
            ..Default::default()
        };
        state.load_witness(witness);

        let backend = Halo2ConstraintCompiler::<C>::new(dsl_operations.num_public_values);
        #[cfg(feature = "bench-metrics")]
        let backend = if collect_metrics {
            backend.with_collect_metrics()
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
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Halo2ProvingPinning {
        let builder = Self::builder(CircuitBuilderStage::Keygen, k);
        let mut builder = Self::populate(builder, dsl_operations, witness, true);
        builder.calculate_params(Some(20));

        let params = read_params(k as u32);
        // let break_points;
        // // if pk already exists, read break points from file
        // let pk = if Path::new("halo2_final.pk").exists() {
        //     let file = File::open("halo2_final.json").unwrap();
        //     break_points = serde_json::from_reader(file).unwrap();
        //     gen_pk(&params, &builder, Some(Path::new("halo2_final.pk")))
        // } else {
        //
        //     pk
        // };
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let pk = keygen_pk2(params.as_ref(), &builder, false).unwrap();
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_keygen_time_ms").set(start.elapsed().as_millis() as f64);
        let break_points = builder.break_points();

        let config_params = builder.config_params.clone();
        let num_pvs = builder
            .assigned_instances
            .iter()
            .map(|x| x.len())
            .collect_vec();

        // let file = File::create("halo2_final.json").unwrap();
        // serde_json::to_writer(file, &break_points).unwrap();
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
        config_params: BaseCircuitParams,
        break_points: MultiPhaseThreadBreakPoints,
        pk: &ProvingKey<G1Affine>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Snark {
        let k = config_params.k;
        let params = read_params(k as u32);
        Self::prove_with_loaded_params(
            &params,
            config_params,
            break_points,
            pk,
            dsl_operations,
            witness,
        )
    }

    pub fn prove_with_loaded_params<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        params: &Halo2Params,
        config_params: BaseCircuitParams,
        break_points: MultiPhaseThreadBreakPoints,
        pk: &ProvingKey<G1Affine>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Snark {
        let k = config_params.k;
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let builder = Self::builder(CircuitBuilderStage::Prover, k)
            .use_params(config_params)
            .use_break_points(break_points);
        let builder = Self::populate(builder, dsl_operations, witness, false);
        #[cfg(feature = "bench-metrics")]
        {
            let stats = builder.statistics();
            let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
            let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
            let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
            metrics::gauge!("halo2_total_cells").set(total_cell as f64);
        }
        let snark = gen_snark_shplonk(params, pk, builder, None::<&str>);

        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_proof_time_ms").set(start.elapsed().as_millis() as f64);

        snark
    }

    pub fn full_prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Snark {
        let Halo2ProvingPinning { pk, metadata } =
            Self::keygen(k, dsl_operations.clone(), witness.clone());
        Self::prove(
            metadata.config_params,
            metadata.break_points,
            &pk,
            dsl_operations,
            witness,
        )
    }
}

fn pk_serializer<S>(value: &ProvingKey<G1Affine>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_bytes(&value.to_bytes(SerdeFormat::RawBytes))
}

impl<'de> Deserialize<'de> for Halo2ProvingPinning {
    fn deserialize<D>(deserializer: D) -> Result<Halo2ProvingPinning, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Pk,
            Metadata,
        }

        struct Halo2ProvingPinningVisitor;

        impl<'de> Visitor<'de> for Halo2ProvingPinningVisitor {
            type Value = Halo2ProvingPinning;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a struct named Halo2ProvingPinning")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut pk_bytes: Option<Vec<u8>> = None;
                let mut metadata: Option<Halo2ProvingMetadata> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Pk => {
                            pk_bytes = Some(map.next_value()?);
                        }
                        Field::Metadata => {
                            metadata = Some(map.next_value()?);
                        }
                    }
                }

                let pk_bytes = pk_bytes.ok_or_else(|| de::Error::missing_field("pk"))?;
                let metadata = metadata.ok_or_else(|| de::Error::missing_field("metadata"))?;
                let pk = ProvingKey::<G1Affine>::from_bytes::<BaseCircuitBuilder<Fr>>(
                    &pk_bytes,
                    SerdeFormat::RawBytes,
                    metadata.config_params.clone(),
                )
                .map_err(|e| de::Error::custom(format!("invalid bytes for proving key: {}", e)))?;

                Ok(Halo2ProvingPinning { pk, metadata })
            }
        }

        deserializer.deserialize_struct(
            "Halo2ProvingPinning",
            &["pk", "metadata"],
            Halo2ProvingPinningVisitor,
        )
    }
}
