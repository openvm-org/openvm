#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
use openvm_continuations::{
    static_verifier::{StaticVerifierConfig, StaticVerifierPvHandler},
    verifier::root::types::RootVmVerifierInput,
};
#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
use openvm_native_compiler::prelude::*;
#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
use openvm_native_recursion::{
    halo2::{verifier::Halo2VerifierProvingKey, Halo2Params, Halo2Prover},
    hints::Hintable,
    witness::Witnessable,
};
#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
use openvm_stark_sdk::openvm_stark_backend::{p3_field::FieldAlgebra, proof::Proof};

#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
use crate::{
    keygen::RootVerifierProvingKey,
    prover::{vm::SingleSegmentVmProver, RootVerifierLocalProver},
    RootSC, F, SC,
};

#[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
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

    #[cfg(any(feature = "evm-prove", feature = "evm-verify"))]
    pub fn generate_dummy_root_proof(&self, dummy_internal_proof: Proof<SC>) -> Proof<RootSC> {
        let prover = RootVerifierLocalProver::new(self.clone());
        // 2 * DIGEST_SIZE for exe_commit and leaf_commit
        let num_public_values = prover
            .root_verifier_pk
            .vm_pk
            .vm_config
            .system
            .num_public_values
            - 2 * DIGEST_SIZE;
        SingleSegmentVmProver::prove(
            &prover,
            RootVmVerifierInput {
                proofs: vec![dummy_internal_proof],
                public_values: vec![F::ZERO; num_public_values],
            }
            .write(),
        )
    }
}
