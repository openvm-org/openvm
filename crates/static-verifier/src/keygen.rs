#[cfg(feature = "halo2-gpu")]
use std::sync::{Arc, Mutex, OnceLock};

use halo2_base::{
    gates::circuit::CircuitBuilderStage,
    halo2_proofs::plonk::{keygen_pk, keygen_vk},
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::proof::Proof,
};
#[cfg(feature = "evm-prove")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "halo2-gpu")]
use crate::graph_executor::GraphProver;
use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    prover::{Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierProof},
};

impl StaticVerifierCircuit {
    /// Run keygen to produce a [`Halo2ProvingPinning`].
    ///
    /// The `representative_proof` is used as a witness for keygen; any valid proof for this static
    /// circuit shape will do.
    pub fn keygen(
        &self,
        params: &Halo2Params,
        shape: &StaticVerifierShape,
        representative_proof: &Proof<RootConfig>,
    ) -> Halo2ProvingPinning {
        let mut builder = Self::builder(CircuitBuilderStage::Keygen, shape);
        self.populate(&mut builder, representative_proof);

        let config_params = builder.calculate_params(Some(shape.minimum_rows));

        let vk = keygen_vk(params, &builder).expect("keygen_vk should succeed");
        let pk = keygen_pk(params, vk, &builder).expect("keygen_pk should succeed");
        let break_points = builder.break_points();

        Halo2ProvingPinning {
            pk,
            metadata: Halo2ProvingMetadata {
                config_params,
                break_points,
                num_pvs: builder
                    .assigned_instances
                    .iter()
                    .map(|instances| instances.len())
                    .collect(),
            },
        }
    }
}

/// High-level proving key that owns a [`StaticVerifierCircuit`], [`Halo2ProvingPinning`], and
/// [`StaticVerifierShape`].
#[derive(Clone)]
pub struct StaticVerifierProvingKey {
    pub circuit: StaticVerifierCircuit,
    pub pinning: Halo2ProvingPinning,
    pub shape: StaticVerifierShape,
    /// Graph-based witness generator used by `prove_wrapped`; built eagerly at
    /// keygen time and reused across proofs (the populate trace only depends on the
    /// static circuit shape). Decoded proving keys start empty and rebuild lazily on
    /// the first call to `prove_wrapped`.
    #[cfg(feature = "halo2-gpu")]
    pub graph_prover: Arc<OnceLock<Mutex<GraphProver>>>,
}

impl StaticVerifierProvingKey {
    /// Run keygen and return a proving key that can be reused for multiple proofs.
    pub fn keygen(
        params: &Halo2Params,
        shape: StaticVerifierShape,
        circuit: StaticVerifierCircuit,
        representative_proof: &Proof<RootConfig>,
    ) -> Self {
        let pinning = circuit.keygen(params, &shape, representative_proof);
        #[cfg(feature = "halo2-gpu")]
        let graph_prover = tracing::info_span!("build_graph_prover").in_scope(|| {
            Arc::new(OnceLock::from(Mutex::new(GraphProver::new(
                &circuit,
                shape.lookup_bits,
                representative_proof,
            ))))
        });
        Self {
            circuit,
            pinning,
            shape,
            #[cfg(feature = "halo2-gpu")]
            graph_prover,
        }
    }

    /// Generate a proof using the stored pinning and shape.
    pub fn prove(&self, params: &Halo2Params, proof: &Proof<RootConfig>) -> StaticVerifierProof {
        self.circuit
            .prove(params, &self.pinning, &self.shape, proof)
    }

    /// Verify a proof against this proving key's verifying key.
    pub fn verify(&self, params: &Halo2Params, proof: &StaticVerifierProof) -> bool {
        StaticVerifierCircuit::verify(params, self.pinning.pk.get_vk(), proof)
    }
}

// --- EVM support (feature-gated) ---

#[cfg(feature = "evm-prove")]
use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder, halo2_proofs::halo2curves::bn256::Fr,
};
#[cfg(feature = "evm-prove")]
use snark_verifier_sdk::{
    evm::{gen_evm_proof_shplonk, gen_evm_verifier_sol_code},
    SHPLONK,
};

/// EVM-compatible proof consisting of instances and raw proof bytes.
#[cfg(feature = "evm-prove")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvmProof {
    pub instances: Vec<Fr>,
    pub proof: Vec<u8>,
}

#[cfg(feature = "evm-prove")]
impl StaticVerifierProvingKey {
    /// Generate a Solidity verifier contract for this circuit.
    pub fn generate_fallback_evm_verifier(&self, params: &Halo2Params) -> String {
        gen_evm_verifier_sol_code::<BaseCircuitBuilder<Fr>, SHPLONK>(
            params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
        )
    }

    /// Produce a [`Snark`](snark_verifier_sdk::Snark) for consumption by the wrapper circuit.
    ///
    /// Unlike [`prove_for_evm_unwrapped`](Self::prove_for_evm_unwrapped), this
    /// returns a `Snark` (not a raw EVM proof), which should be fed into
    /// [`Halo2WrapperProvingKey::prove_for_evm`](crate::wrapper::Halo2WrapperProvingKey::prove_for_evm).
    ///
    /// Witness generation runs through the [`GraphProver`] (parallel graph-IR
    /// evaluation) instead of the halo2 `BaseCircuitBuilder`; the raw advice and
    /// range-check tapes are then laid out into physical columns and copied to device
    /// for [`gen_snark_from_base`](snark_verifier_sdk::halo2::gen_snark_from_base).
    ///
    /// The graph executor thread count defaults to available cores − 2; override
    /// with `GRAPH_EXE_THREADS`. Set `STATIC_VERIFIER_COMPARE_WITNESS=1` to
    /// additionally regenerate the witness through the legacy `BaseCircuitBuilder` +
    /// `synthesize_witness_shplonk` path and assert both advice layouts match.
    #[cfg(feature = "halo2-gpu")]
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> snark_verifier_sdk::Snark {
        // Cores − 2 leaves one core for the release-walk callback and one for
        // the rest of the runtime (GPU driver threads, tokio, etc.).
        let graph_prover_threads: usize = std::env::var("GRAPH_EXE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&t: &usize| t > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get().saturating_sub(2).max(1))
                    .unwrap_or(1)
            });

        let (gpu_advice, instances) =
            self.run_witness_gen_pipeline(proof, graph_prover_threads, Some(params));

        snark_verifier_sdk::halo2::gen_snark_from_base(
            params,
            &self.pinning.pk,
            gpu_advice,
            instances,
        )
    }

    /// Produce a [`Snark`](snark_verifier_sdk::Snark) for consumption by the wrapper circuit.
    ///
    /// CPU witness generation through the halo2 `BaseCircuitBuilder`; enable the
    /// `halo2-gpu` feature for the parallel graph-IR + GPU pipeline.
    #[cfg(not(feature = "halo2-gpu"))]
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> snark_verifier_sdk::Snark {
        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);

        let _public_inputs = self.circuit.populate(&mut builder, proof);

        snark_verifier_sdk::halo2::gen_snark_shplonk(
            params,
            &self.pinning.pk,
            builder,
            None::<&str>,
        )
    }

    /// Runs the graph-executor witness pipeline: builds (or reuses) the
    /// [`GraphProver`], streams its advice/lookup deltas through a
    /// [`FusedColumnBuilder`](crate::graph_executor::FusedColumnBuilder) onto device
    /// columns, and returns the `Vec<DeviceBuffer<Fr>>` + instance columns ready for
    /// [`gen_snark_from_base`](snark_verifier_sdk::halo2::gen_snark_from_base).
    ///
    /// Public so benchmarks (see the `graph_executor_prove_wrapped_pipeline` test)
    /// can time the witness path in isolation from SNARK generation.
    ///
    /// `diagnostic_params` is only consulted when `STATIC_VERIFIER_COMPARE_WITNESS`
    /// is set — see [`Self::compare_witness_with_base_builder`].
    #[cfg(feature = "halo2-gpu")]
    pub fn run_witness_gen_pipeline(
        &self,
        proof: &Proof<RootConfig>,
        num_threads: usize,
        diagnostic_params: Option<&Halo2Params>,
    ) -> (
        Vec<halo2_base::halo2_proofs::cuda::DeviceBuffer<Fr>>,
        Vec<Vec<Fr>>,
    ) {
        use halo2_base::{
            gates::circuit::MaybeRangeConfig,
            halo2_proofs::{halo2curves::bn256::G1Affine, plonk::create_constraint_system},
        };
        use tracing::info_span;

        use crate::graph_executor::FusedColumnBuilder;

        let graph_prover = self.graph_prover.get_or_init(|| {
            info_span!("build_graph_prover_lazy").in_scope(|| {
                Mutex::new(GraphProver::new(
                    &self.circuit,
                    self.shape.lookup_bits,
                    proof,
                ))
            })
        });
        let mut graph_prover = graph_prover.lock().unwrap();

        // Pre-derive the physical column layout that the fused closure will fill.
        let num_advice_columns = self.pinning.pk.get_vk().cs().num_advice_columns();
        let n = 1usize << self.pinning.metadata.config_params.k;
        let (_, config) = create_constraint_system::<G1Affine, BaseCircuitBuilder<Fr>>(
            self.pinning.metadata.config_params.clone(),
        );
        let MaybeRangeConfig::WithRange(range_config) = &config.base else {
            panic!("static verifier requires lookup advice columns");
        };
        let lookup_col_indices: Vec<usize> = range_config.lookup_advice[0]
            .iter()
            .map(|c| c.index())
            .collect();
        assert!(
            !lookup_col_indices.is_empty(),
            "range lookups require lookup advice columns"
        );
        let max_lookup_rows = range_config.gate.max_rows;
        let break_points = self.pinning.metadata.break_points[0].clone();
        assert!(
            graph_prover
                .total_lookup_cells()
                .div_ceil(lookup_col_indices.len())
                <= max_lookup_rows,
            "range lookups would be assigned to unusable rows"
        );

        let mut builder =
            FusedColumnBuilder::new(n, num_advice_columns, break_points, lookup_col_indices);

        let pvs = info_span!("graph_witness_gen", num_threads).in_scope(|| {
            graph_prover.witness_gen(
                &self.circuit,
                proof,
                num_threads,
                |advice_offset, advice_delta, lookup_offset, lookup_delta| {
                    builder.append(advice_offset, advice_delta, lookup_offset, lookup_delta)
                },
            )
        });
        drop(graph_prover);

        let mut instances = vec![Vec::new(); self.shape.instance_columns];
        instances[0] = pvs;

        if let Some(params) = diagnostic_params {
            if std::env::var("STATIC_VERIFIER_COMPARE_WITNESS").is_ok() {
                let host_columns = builder.snapshot_columns_to_host();
                self.compare_witness_with_base_builder(params, proof, &host_columns, &instances);
            }
        }

        let gpu_advice = builder.take_device_columns();
        (gpu_advice, instances)
    }

    /// Diagnostic: regenerates the witness through the legacy `BaseCircuitBuilder`
    /// populate + `synthesize_witness_shplonk` path and asserts the post-break-point,
    /// post-lookup advice matches the graph-executor path. Blinding rows are excluded
    /// (the synthesize path randomizes them; the materialized path leaves them zero).
    #[cfg(feature = "halo2-gpu")]
    fn compare_witness_with_base_builder(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
        new_columns: &[Vec<Fr>],
        new_instances: &[Vec<Fr>],
    ) {
        use halo2_base::halo2_proofs::poly::DevicePolyExt;
        use tracing::info;

        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);
        let _public_inputs = self.circuit.populate(&mut builder, proof);
        let instances_old: Vec<Vec<Fr>> = builder
            .assigned_instances
            .iter()
            .map(|column| column.iter().map(|v| *v.value()).collect())
            .collect();
        assert_eq!(
            new_instances,
            &instances_old[..],
            "instances differ from BaseCircuitBuilder path"
        );

        let (_, mut advice, _) = snark_verifier_sdk::halo2::synthesize_witness_shplonk(
            params,
            &self.pinning.pk,
            builder,
            instances_old,
        );
        let advice = advice.remove(0);
        let usable_rows = (1usize << self.pinning.metadata.config_params.k)
            - (self.pinning.pk.get_vk().cs().blinding_factors() + 1);
        assert_eq!(
            advice.advice_values.len(),
            new_columns.len(),
            "advice column count differs"
        );
        for (col, (device_poly, new_column)) in
            advice.advice_values.iter().zip(new_columns).enumerate()
        {
            let host_poly = device_poly.to_host();
            let old_column = host_poly.values();
            let mut diffs = 0usize;
            for row in 0..usable_rows {
                if old_column[row] != new_column[row] {
                    eprintln!(
                        "advice col {col} row {row} differs: old={:?} new={:?}",
                        old_column[row], new_column[row]
                    );
                    diffs += 1;
                    assert!(diffs < 20, "too many diffs in advice col {col}");
                }
            }
            assert_eq!(
                diffs, 0,
                "advice col {col} differs from synthesize_witness_shplonk"
            );
        }
        info!("graph-executor advice matches BaseCircuitBuilder + synthesize_witness_shplonk");
    }

    /// Generate a dummy snark for wrapper keygen.
    pub fn generate_dummy_snark(
        &self,
        reader: &impl crate::wrapper::Halo2ParamsReader,
    ) -> snark_verifier_sdk::Snark {
        let k = self.pinning.metadata.config_params.k;
        let params = reader.read_params(k);
        snark_verifier_sdk::halo2::gen_dummy_snark_from_vk::<SHPLONK>(
            &params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
            None,
        )
    }

    /// Generate an EVM-compatible proof directly (one-step, no wrapper circuit).
    pub fn prove_for_evm_unwrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> RawEvmProof {
        self.shape.assert_onchain_verifier_supported();

        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);

        let public_inputs = self.circuit.populate(&mut builder, proof);
        let instances_vec = public_inputs.to_vec();

        let snark = gen_evm_proof_shplonk(
            params,
            &self.pinning.pk,
            builder,
            vec![instances_vec.clone()],
        );

        RawEvmProof {
            instances: instances_vec,
            proof: snark,
        }
    }
}

/// Verify an EVM proof using a deployed verifier contract.
///
/// Returns the gas used on success, or an error message on failure.
#[cfg(feature = "evm-verify")]
pub fn evm_verify(deployment_code: &[u8], proof: &RawEvmProof) -> Result<u64, String> {
    snark_verifier_sdk::evm::evm_verify(
        deployment_code.to_vec(),
        vec![proof.instances.clone()],
        proof.proof.clone(),
    )
    .map_err(|e| format!("EVM verification failed: {e}"))
}
