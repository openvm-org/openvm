#[cfg(feature = "evm-prove")]
use std::io::Write;

use halo2_base::{
    gates::circuit::CircuitBuilderStage,
    halo2_proofs::{
        plonk::{keygen_pk, keygen_vk},
        poly::commitment::Params,
    },
    ContextKind, DummyContext,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::proof::Proof,
};
#[cfg(feature = "evm-prove")]
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    prover::{Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierProof},
};

#[cfg(feature = "evm-prove")]
struct PerfCtlGuard {
    pipe: Option<std::fs::File>,
}

#[cfg(feature = "evm-prove")]
impl PerfCtlGuard {
    fn enable() -> Self {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("perf.ctl");
        let pipe = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .ok()
            .and_then(|mut pipe| {
                if pipe
                    .write_all(b"enable\n")
                    .and_then(|_| pipe.flush())
                    .is_ok()
                {
                    Some(pipe)
                } else {
                    None
                }
            });

        Self { pipe }
    }
}

#[cfg(feature = "evm-prove")]
impl Drop for PerfCtlGuard {
    fn drop(&mut self) {
        if let Some(pipe) = &mut self.pipe {
            let _ = pipe.write_all(b"disable\n").and_then(|_| pipe.flush());
        }
    }
}

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
        let col_size = vk.cs().num_advice_columns() * (params.n() as usize);
        let mut pk = keygen_pk(params, vk, &builder).expect("keygen_pk should succeed");
        let ctx = builder.main(0);
        assert!(col_size >= ctx.get_offset());
        // let copy_manager = ctx.copy_manager.lock().unwrap();
        // pk.perf_hints.advice_equalities = Some(copy_manager.advice_equalities.len());
        // pk.perf_hints.constant_equalities = Some(copy_manager.advice_equalities.len());
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
        Self {
            circuit,
            pinning,
            shape,
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
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> snark_verifier_sdk::Snark {
        use halo2_base::{
            gates::circuit::builder::WitnessCircuitBuilder,
            halo2_proofs::{
                halo2curves::bn256::{Fr, G1Affine},
                plonk::{create_constraint_system, AdviceSingle, InstanceSingle},
                poly::DevicePolyExt,
            },
        };
        use tracing::info_span;

        // --- Diagnostic comparison: OLD (BaseCircuitBuilder + populate → Circuit::synthesize)
        // vs NEW (WitnessCircuitBuilder + populate_witness_gen + materialize).
        // Both paths should yield identical (instance_single, advice_single, challenges).
        // let (i_old, a_old, c_old) = {
        //     let mut base_builder = BaseCircuitBuilder::prover(
        //         self.pinning.metadata.config_params.clone(),
        //         self.pinning.metadata.break_points.clone(),
        //     )
        //     .use_instance_columns(self.shape.instance_columns);
        //     let _ = self.circuit.populate(&mut base_builder, proof);
        //     let instances_old: Vec<Vec<Fr>> = base_builder
        //         .assigned_instances
        //         .iter()
        //         .map(|column| column.iter().map(|av| *av.value()).collect())
        //         .collect();
        //     let (mut i, mut a, c) = snark_verifier_sdk::halo2::synthesize_witness_shplonk(
        //         params,
        //         &self.pinning.pk,
        //         base_builder,
        //         instances_old,
        //     );
        //     (i.remove(0), a.remove(0), c)
        // };
        //
        // let (i_new, a_new, c_new) = {
        //     let mut wit_builder_diag = WitnessCircuitBuilder::new(
        //         self.pinning.metadata.break_points[0].clone(),
        //         self.pinning.metadata.config_params.clone(),
        //         self.pinning.pk.get_vk().cs().num_advice_columns(),
        //     );
        //     self.circuit
        //         .populate_witness_gen(&mut wit_builder_diag, proof);
        //     let (_, config) = create_constraint_system::<G1Affine, BaseCircuitBuilder<Fr>>(
        //         self.pinning.metadata.config_params.clone(),
        //     );
        //     wit_builder_diag.assign_lookups_to_advice(&config, 0);
        //     snark_verifier_sdk::halo2::materialize_witness_shplonk(
        //         params,
        //         &self.pinning.pk,
        //         wit_builder_diag,
        //     )
        // };
        //
        // assert_eq!(c_old.len(), c_new.len(), "challenges differ in length");
        // for i in 0..c_old.len() {
        //     assert_eq!(c_old[i], c_new[i], "challenges differ in {i}")
        // }
        // assert_instance_single_eq(&i_old, &i_new);
        // assert_advice_single_eq(&a_old, &a_new);
        // info!("OLD vs NEW witness comparison: match");

        // --- Diagnostic: measure the pure witness-generation code cost with
        // no advice buffer writes or GPU traffic. Builder construction is kept
        // out of the span so only populate_ctx is timed.
        // {
        //     use crate::field::baby_bear::{BabyBearChip, BabyBearExtChip};
        //     use std::sync::Arc;
        //
        //     let dummy_builder = WitnessCircuitBuilder::<Fr>::new(
        //         self.pinning.metadata.break_points[0].clone(),
        //         self.pinning.metadata.config_params.clone(),
        //         self.pinning.pk.get_vk().cs().num_advice_columns(),
        //     );
        //     let range = dummy_builder.range_chip();
        //     let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
        //     let mut dummy = DummyContext::<Fr>::new();
        //     info_span!("dummy witness-gen")
        //         .in_scope(|| self.circuit.populate_ctx(&mut dummy, &ext_chip, proof));
        // }

        // --- Actual proving via NEW path.
        let mut builder = WitnessCircuitBuilder::new(
            self.pinning.metadata.break_points[0].clone(),
            self.pinning.metadata.config_params.clone(),
            self.pinning.pk.get_vk().cs().num_advice_columns(),
        );

        info_span!("witness_gen").in_scope(|| {
            let _perf_ctl = PerfCtlGuard::enable();
            self.circuit.populate_witness_gen(&mut builder, proof);
        });
        let (_, config) = create_constraint_system::<G1Affine, BaseCircuitBuilder<Fr>>(
            self.pinning.metadata.config_params.clone(),
        );

        info_span!("assign_lookups_to_advice")
            .in_scope(|| builder.assign_lookups_to_advice(&config, 0));
        info!("advice_len {}", builder.main().get_offset());

        snark_verifier_sdk::halo2::gen_snark_from_witness(params, &self.pinning.pk, builder)
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

#[cfg(feature = "evm-prove")]
fn assert_column_values_eq<F: PartialEq + std::fmt::Debug>(
    kind: &str,
    col: usize,
    a: &[F],
    b: &[F],
) {
    assert_eq!(
        a.len(),
        b.len(),
        "{kind} column {col} length differs: old={} new={}",
        a.len(),
        b.len(),
    );
    let mut diff_count = 0;
    for (row, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        if va != vb {
            diff_count += 1;
            eprintln!("{kind} column {col} row {row} differs: old={va:?} new={vb:?}");
            if diff_count >= 20 {
                assert!(
                    va == vb,
                    "{kind} column {col} row {row} differs: old={va:?} new={vb:?}",
                );
            }
        }
    }
}

#[cfg(feature = "evm-prove")]
fn assert_instance_single_eq(
    a: &halo2_base::halo2_proofs::plonk::InstanceSingle<
        halo2_base::halo2_proofs::halo2curves::bn256::G1Affine,
    >,
    b: &halo2_base::halo2_proofs::plonk::InstanceSingle<
        halo2_base::halo2_proofs::halo2curves::bn256::G1Affine,
    >,
) {
    use halo2_base::halo2_proofs::poly::DevicePolyExt;
    assert_eq!(
        a.instance_values.len(),
        b.instance_values.len(),
        "instance_values column count differs"
    );
    for (col, (pa, pb)) in a
        .instance_values
        .iter()
        .zip(b.instance_values.iter())
        .enumerate()
    {
        let ha = pa.to_host();
        let hb = pb.to_host();
        assert_column_values_eq("instance_values", col, ha.values(), hb.values());
    }
    assert_eq!(
        a.instance_polys.len(),
        b.instance_polys.len(),
        "instance_polys column count differs"
    );
    for (col, (pa, pb)) in a
        .instance_polys
        .iter()
        .zip(b.instance_polys.iter())
        .enumerate()
    {
        let ha = pa.to_host();
        let hb = pb.to_host();
        assert_column_values_eq("instance_polys", col, ha.values(), hb.values());
    }
}

#[cfg(feature = "evm-prove")]
fn assert_advice_single_eq(
    a: &halo2_base::halo2_proofs::plonk::AdviceSingle<
        halo2_base::halo2_proofs::halo2curves::bn256::G1Affine,
    >,
    b: &halo2_base::halo2_proofs::plonk::AdviceSingle<
        halo2_base::halo2_proofs::halo2curves::bn256::G1Affine,
    >,
) {
    use halo2_base::halo2_proofs::poly::DevicePolyExt;
    assert_eq!(
        a.advice_values.len(),
        b.advice_values.len(),
        "advice_values column count differs"
    );
    for (col, (pa, pb)) in a
        .advice_values
        .iter()
        .zip(b.advice_values.iter())
        .enumerate()
    {
        let ha = pa.to_host();
        let hb = pb.to_host();
        assert_column_values_eq("advice_values", col, ha.values(), hb.values());
    }
    assert_eq!(
        a.advice_polys.len(),
        b.advice_polys.len(),
        "advice_polys column count differs"
    );
    for (col, (pa, pb)) in a.advice_polys.iter().zip(b.advice_polys.iter()).enumerate() {
        let ha = pa.to_host();
        let hb = pb.to_host();
        assert_column_values_eq("advice_polys", col, ha.values(), hb.values());
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
