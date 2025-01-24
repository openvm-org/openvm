use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{hints::Hintable, vars::StarkProofVariable};
use openvm_stark_sdk::openvm_stark_backend::{config::Val, proof::Proof};

use crate::{verifier::root::types::RootVmVerifierInput, C, SC};

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
