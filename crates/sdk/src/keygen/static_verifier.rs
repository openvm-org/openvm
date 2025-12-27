use openvm_circuit::arch::{SingleSegmentVmProver, VirtualMachineError};
use openvm_continuations::{
    static_verifier::{StaticVerifierConfig, StaticVerifierPvHandler},
    verifier::root::types::RootVmVerifierInput,
};
use openvm_native_circuit::NATIVE_MAX_TRACE_HEIGHTS;
use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{
    halo2::{verifier::Halo2VerifierProvingKey, Halo2Params, Halo2Prover},
    hints::Hintable,
    witness::Witnessable,
};
use openvm_stark_sdk::openvm_stark_backend::{p3_field::FieldAlgebra, proof::Proof};

use crate::{keygen::RootVerifierProvingKey, prover::RootVerifierLocalProver, RootSC, F, SC};

impl RootVerifierProvingKey {
    /// Keygen the static verifier for this root verifier.
    pub fn keygen_static_verifier(
        &self,
        params: &Halo2Params,
        root_proof: Proof<RootSC>,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> Halo2VerifierProvingKey {
        let mut witness = Witness::default();
        root_proof.write(&mut witness);
        let special_air_ids = self.air_id_permutation().get_special_air_ids();
        let config = StaticVerifierConfig {
            root_verifier_fri_params: self.vm_pk.fri_params,
            special_air_ids,
            root_verifier_program_commit: self.root_committed_exe.get_program_commit().into(),
        };
        let dsl_operations = config.build_static_verifier_operations(
            &self.vm_pk.vm_pk.get_vk(),
            &root_proof,
            pv_handler,
        );
        Halo2VerifierProvingKey {
            pinning: Halo2Prover::keygen(params, dsl_operations.clone(), witness),
            dsl_ops: dsl_operations,
        }
    }

    pub fn generate_dummy_root_proof(
        &self,
        dummy_internal_proof: Proof<SC>,
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        let mut prover = RootVerifierLocalProver::new(self)?;
        // 2 * DIGEST_SIZE for exe_commit and leaf_commit
        let num_public_values = prover.vm_config().as_ref().num_public_values - 2 * DIGEST_SIZE;
        SingleSegmentVmProver::prove(
            &mut prover,
            RootVmVerifierInput {
                proofs: vec![dummy_internal_proof],
                public_values: vec![F::ZERO; num_public_values],
            }
            .write(),
            NATIVE_MAX_TRACE_HEIGHTS,
        )
    }
}
