use std::array;

use ax_stark_sdk::{
    ax_stark_backend::prover::types::Proof, config::baby_bear_poseidon2::BabyBearPoseidon2Config,
};
use axvm_native_compiler::{ir::DIGEST_SIZE, prelude::*};
use axvm_recursion::{hints::Hintable, types::InnerConfig, vars::StarkProofVariable};

use crate::verifier::{internal::types::InternalVmVerifierInput, utils::write_field_slice};

#[derive(DslVariable, Clone)]
pub struct InternalVmVerifierInputVariable<C: Config> {
    pub self_program_commit: [Felt<C::F>; DIGEST_SIZE],
    /// The proofs of the execution segments in the execution order.
    pub proofs: Array<C, StarkProofVariable<C>>,
}

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;

impl Hintable<C> for InternalVmVerifierInput<SC> {
    type HintVariable = InternalVmVerifierInputVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let self_program_commit = array::from_fn(|_| builder.hint_felt());
        let proofs = Vec::<Proof<SC>>::read(builder);
        Self::HintVariable {
            self_program_commit,
            proofs,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = write_field_slice(&self.self_program_commit);
        stream.extend(self.proofs.write());
        stream
    }
}
