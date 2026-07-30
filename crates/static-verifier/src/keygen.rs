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

use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    prover::{Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierProof},
    tracegen::graph_executor::GraphProgram,
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
    pub graph_program: GraphProgram,
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
        let graph_program = tracing::info_span!("build_graph_program")
            .in_scope(|| GraphProgram::new(&circuit, shape.lookup_bits, representative_proof));
        Self {
            circuit,
            pinning,
            shape,
            graph_program,
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
    gates::circuit::builder::BaseCircuitBuilder,
    halo2_proofs::{halo2curves::bn256::Fr, plonk::AdviceColumns},
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
    /// `state` is the caller-owned witness scratch (see
    /// [`GraphExecutorState`](crate::tracegen::graph_executor::GraphExecutorState));
    /// reusing it across proofs avoids reallocating the tape and flag buffers.
    ///
    /// Unlike [`prove_for_evm_unwrapped`](Self::prove_for_evm_unwrapped), this
    /// returns a `Snark` (not a raw EVM proof), which should be fed into
    /// [`Halo2WrapperProvingKey::prove_for_evm`](crate::wrapper::Halo2WrapperProvingKey::prove_for_evm).
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
        state: &mut crate::tracegen::graph_executor::GraphExecutorState,
    ) -> snark_verifier_sdk::Snark {
        // Cores − 2 leaves one core for the release-walk callback and one for
        // the rest of the runtime (GPU driver threads, tokio, etc.).
        static GRAPH_EXE_THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let num_threads = *GRAPH_EXE_THREADS.get_or_init(|| {
            std::env::var("GRAPH_EXE_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&t: &usize| t > 0)
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get().saturating_sub(2).max(1))
                        .unwrap_or(1)
                })
        });

        let (advice, instances) = self.generate_witness(proof, num_threads, state);

        snark_verifier_sdk::halo2::gen_snark_from_base(params, &self.pinning.pk, advice, instances)
    }

    /// Runs the graph-executor witness pipeline: binds the stored
    /// [`GraphProgram`] to `state`, streams the advice/lookup deltas through a
    /// [`FusedColumnBuilder`](crate::tracegen::graph_executor::FusedColumnBuilder) onto advice
    /// columns, and returns the [`AdviceColumns<Fr>`] + instance columns ready for
    /// [`gen_snark_from_base`](snark_verifier_sdk::halo2::gen_snark_from_base).
    pub fn generate_witness(
        &self,
        proof: &Proof<RootConfig>,
        num_threads: usize,
        state: &mut crate::tracegen::graph_executor::GraphExecutorState,
    ) -> (AdviceColumns<Fr>, Vec<Vec<Fr>>) {
        use halo2_base::{
            gates::circuit::MaybeRangeConfig,
            halo2_proofs::plonk::{Circuit, ConstraintSystem},
        };
        use tracing::info_span;

        use crate::{
            stages::full_pipeline::load_proof_wire,
            tracegen::graph_executor::{FusedColumnBuilder, GraphExecutor},
        };

        // Pre-derive the physical column layout that the fused closure will fill.
        let num_advice_columns = self.pinning.pk.get_vk().cs().num_advice_columns();
        let n = 1usize << self.pinning.metadata.config_params.k;
        let mut cs = ConstraintSystem::<Fr>::default();
        let config = <BaseCircuitBuilder<Fr> as Circuit<Fr>>::configure_with_params(
            &mut cs,
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
            self.graph_program
                .lookup_cells()
                .div_ceil(lookup_col_indices.len())
                <= max_lookup_rows,
            "range lookups would be assigned to unusable rows"
        );

        let mut builder =
            FusedColumnBuilder::new(n, num_advice_columns, break_points, lookup_col_indices);

        let mut executor = GraphExecutor::new(&self.graph_program, state);
        executor.state.reset();
        info_span!("populate_inputs").in_scope(|| {
            load_proof_wire(&mut executor, proof, &self.circuit.log_heights_per_air);
        });
        info_span!("graph_witness_gen", num_threads).in_scope(|| {
            executor.run(
                num_threads,
                |advice_offset, advice_delta, lookup_offset, lookup_delta| {
                    builder.append(advice_offset, advice_delta, lookup_offset, lookup_delta)
                },
            );
        });
        let pvs = info_span!("collect_pvs").in_scope(|| {
            let advice = executor.advice();
            self.graph_program
                .pv_offsets()
                .iter()
                .map(|&offset| advice[offset])
                .collect()
        });

        let mut instances = vec![Vec::new(); self.shape.instance_columns];
        instances[0] = pvs;

        let advice = builder.take_columns();
        (advice, instances)
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
