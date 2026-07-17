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

use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    graph_executor::GraphProver,
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
    /// Graph-based witness generator used by [`Self::prove_wrapped`]; built eagerly at
    /// keygen time and reused across proofs (the populate trace only depends on the
    /// static circuit shape). Decoded proving keys start empty and rebuild lazily on
    /// the first call to [`Self::prove_wrapped`].
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
        let graph_prover = Arc::new(OnceLock::new());
        tracing::info_span!("build_graph_prover").in_scope(|| {
            graph_prover
                .set(Mutex::new(GraphProver::new(
                    &circuit,
                    shape.lookup_bits,
                    representative_proof,
                )))
                .ok()
                .expect("OnceLock is fresh");
        });
        Self {
            circuit,
            pinning,
            shape,
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
    /// Set `STATIC_VERIFIER_COMPARE_WITNESS=1` to additionally regenerate the witness
    /// through the legacy `BaseCircuitBuilder` + `synthesize_witness_shplonk` path and
    /// assert both advice layouts match.
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> snark_verifier_sdk::Snark {
        // Default to (visible cores − 2) so one core stays free for the callback
        // thread + the rest of the runtime (proof-generation kernels launched
        // from Halo2, tokio driver, etc.); leaves all cores busy without the
        // graph executor stealing from them.
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

    /// Runs the full graph-executor + column-materialization pipeline: builds
    /// (or reuses) the [`GraphProver`], streams the level-by-level advice/lookup
    /// deltas through a [`FusedColumnBuilder`] whose H2D copies land directly on
    /// device columns, and returns the finalized `Vec<DeviceBuffer<Fr>>` +
    /// instance columns ready to hand to
    /// [`gen_snark_from_base`](snark_verifier_sdk::halo2::gen_snark_from_base).
    ///
    /// Exposed as a public entry point so downstream benchmarks (see the
    /// `graph_executor_prove_wrapped_pipeline` `#[ignore]` test) can time the
    /// prover-side witness path in isolation from SNARK generation.
    ///
    /// `diagnostic_params` is only consulted when `STATIC_VERIFIER_COMPARE_WITNESS`
    /// is set — see [`Self::compare_witness_with_base_builder`]. Benchmarks
    /// should pass `None`.
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
                |advice_delta, lookup_delta| builder.append(advice_delta, lookup_delta),
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

/// Streaming builder for the physical advice column layout used by
/// [`StaticVerifierProvingKey::prove_wrapped`].
///
/// The graph executor materializes advice/lookup values level by level; this builder
/// consumes each level's *delta* slice (only the newly-materialized values, in
/// strictly ascending offset order) via [`Self::append`], mirroring
/// `PagedWitnessContext::push_advice` (the gate-column stream splits at pinned break
/// points, duplicating the break-row value at row 0 of the next column so the
/// gate-overlap copy constraint holds) and
/// `BaseCircuitBuilder::assign_lookups_in_phase` (the range-check stream fills lookup
/// advice columns round-robin: value `i` at column `i % L`, row `i / L`).
///
/// Column state lives directly on the GPU: on the first call to [`Self::append`] the
/// builder allocates `num_advice_columns × n` [`DeviceBuffer`]s and zero-fills each
/// (via `cudaMemsetAsync`). Subsequent calls copy each contiguous delta segment
/// straight to the target row range of the target device column with
/// `DeviceBufferExt::mut_slice` + `copy_from_host`, so no large host-side column
/// buffer is ever materialized.
#[cfg(feature = "evm-prove")]
struct FusedColumnBuilder {
    // ---- Config (immutable after `new`) ------------------------------------
    n: usize,
    num_advice_columns: usize,
    /// Pinned break points, in order (consumed as columns split).
    break_points: Vec<usize>,
    /// Physical column indices of the range-check lookup advice columns.
    lookup_col_indices: Vec<usize>,

    // ---- Device columns (lazily allocated on first `append`) ---------------
    device_columns: Vec<halo2_base::halo2_proofs::cuda::DeviceBuffer<Fr>>,

    // ---- Gate-column stream cursor ----------------------------------------
    col: usize,
    row: usize,
    /// Next pinned break-point row (`None` after the final gate column).
    cur_break_point: Option<usize>,
    break_idx: usize,

    // ---- Round-robin lookup stream cursor ---------------------------------
    /// Absolute index of the next lookup value within the round-robin schedule.
    lookup_processed: usize,
}

#[cfg(feature = "evm-prove")]
impl FusedColumnBuilder {
    fn new(
        n: usize,
        num_advice_columns: usize,
        break_points: Vec<usize>,
        lookup_col_indices: Vec<usize>,
    ) -> Self {
        let cur_break_point = break_points.first().copied();
        Self {
            n,
            num_advice_columns,
            break_points,
            lookup_col_indices,
            device_columns: Vec::new(),
            col: 0,
            row: 0,
            cur_break_point,
            break_idx: 0,
            lookup_processed: 0,
        }
    }

    fn ensure_allocated(&mut self) {
        use halo2_base::halo2_proofs::cuda::{utils::HALO2_GPU_CTX, DeviceBuffer};
        if !self.device_columns.is_empty() {
            return;
        }
        self.device_columns.reserve_exact(self.num_advice_columns);
        for _ in 0..self.num_advice_columns {
            let buf: DeviceBuffer<Fr> =
                DeviceBuffer::<Fr>::with_capacity_on(self.n, &HALO2_GPU_CTX);
            buf.fill_zero_on(&HALO2_GPU_CTX)
                .expect("zero-fill advice column");
            self.device_columns.push(buf);
        }
    }

    /// Appends the next contiguous delta of advice and range-check values. Deltas
    /// must arrive in strictly ascending order across successive calls.
    fn append(&mut self, advice_delta: &[Fr], lookup_delta: &[Fr]) {
        use halo2_base::halo2_proofs::cuda::{utils::HALO2_GPU_CTX, DeviceBufferExt as _};

        self.ensure_allocated();

        // --- Gate stream: contiguous H2D per (column, row-range) segment ----
        //
        // Rewind trick: when a segment ends on a break-point row we `delta_pos -= 1`
        // so the break value re-appears as the first host source of the next
        // column's copy — that fills row 0 of the new column "for free" and avoids
        // a separate 1-element H2D.
        let mut delta_pos = 0usize;
        while delta_pos < advice_delta.len() {
            let rows_until_break = match self.cur_break_point {
                Some(bp) => {
                    debug_assert!(bp >= self.row);
                    bp - self.row + 1
                }
                None => usize::MAX,
            };
            let delta_remaining = advice_delta.len() - delta_pos;
            let take = rows_until_break.min(delta_remaining);
            let src = &advice_delta[delta_pos..delta_pos + take];
            self.device_columns[self.col]
                .mut_slice(self.row..self.row + take)
                .copy_from_host(src, &HALO2_GPU_CTX)
                .expect("H2D advice gate segment");
            delta_pos += take;
            if self.cur_break_point.is_some() && take == rows_until_break {
                self.col += 1;
                self.row = 0;
                delta_pos -= 1; // Re-emit the break value as row 0 of the new column.
                self.break_idx += 1;
                self.cur_break_point = self.break_points.get(self.break_idx).copied();
            } else {
                self.row += take;
            }
        }

        // --- Lookup stream: one gathered H2D per lookup column per call ------
        //
        // Value at absolute index `i` lands at column `L[i % L]`, row `i / L`.
        // Within a single delta the indices bound for physical column `L[c]` are
        // strided by `L` in delta space but consecutive in row space. Gather each
        // such stride into a small host buffer (≤ ceil(delta_len / L) elements)
        // and issue one contiguous H2D per lookup column per `append` call.
        let l = self.lookup_col_indices.len();
        let k = self.lookup_processed;
        let n_l = lookup_delta.len();
        if n_l > 0 {
            for c in 0..l {
                let start_j = (c + l - k % l) % l;
                if start_j >= n_l {
                    continue;
                }
                let n_values = (n_l - start_j).div_ceil(l);
                let start_row = (k + start_j) / l;
                let host_buf: Vec<Fr> = (0..n_values)
                    .map(|i| lookup_delta[start_j + i * l])
                    .collect();
                self.device_columns[self.lookup_col_indices[c]]
                    .mut_slice(start_row..start_row + n_values)
                    .copy_from_host(&host_buf, &HALO2_GPU_CTX)
                    .expect("H2D lookup column gather");
            }
        }
        self.lookup_processed += n_l;
    }

    /// Consumes the device columns, leaving the builder empty.
    fn take_device_columns(&mut self) -> Vec<halo2_base::halo2_proofs::cuda::DeviceBuffer<Fr>> {
        assert!(
            !self.device_columns.is_empty(),
            "take_device_columns: no data was ever appended",
        );
        std::mem::take(&mut self.device_columns)
    }

    /// Diagnostic-only: D2H each device column back into host `Vec<Fr>`s so the
    /// caller can byte-compare against the legacy `BaseCircuitBuilder` +
    /// `synthesize_witness_shplonk` path. Not used on the hot prove path.
    fn snapshot_columns_to_host(&self) -> Vec<Vec<Fr>> {
        use halo2_base::halo2_proofs::cuda::{utils::HALO2_GPU_CTX, MemCopyD2H as _};
        self.device_columns
            .iter()
            .map(|d| d.to_host_on(&HALO2_GPU_CTX).expect("D2H advice column"))
            .collect()
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
