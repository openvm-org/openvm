use ax_stark_sdk::{
    ax_stark_backend::{config::Val, prover::types::Proof},
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
};
use axvm_native_compiler::prelude::*;
use axvm_recursion::{hints::Hintable, types::InnerConfig, vars::StarkProofVariable};

use crate::verifier::root::types::RootVmVerifierInput;

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;

#[derive(DslVariable, Clone)]
pub struct RootVmVerifierInputVariable<C: Config> {
    /// The proofs of leaf verifier or internal verifier in the execution order.
    pub proofs: Array<C, StarkProofVariable<C>>,
    /// Public values to expose
    pub public_values: Array<C, Felt<C::F>>,
}

impl Hintable<C> for RootVmVerifierInput<SC> {
    type HintVariable = RootVmVerifierInputVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let proofs = Vec::<Proof<SC>>::read(builder);
        let public_values = Vec::<Val<SC>>::read(builder);
        Self::HintVariable {
            proofs,
            public_values,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = self.proofs.write();
        stream.extend(self.public_values.write());
        stream
    }
}
