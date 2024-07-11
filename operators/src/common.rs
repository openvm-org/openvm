use std::any::Any;

use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::engine::StarkEngine;
use p3_uni_stark::StarkGenericConfig;

pub trait Verifiable<SC: StarkGenericConfig, E: StarkEngine<SC>>: Any {
    // TODO: Although conceptually all circuits structs shouldn't change
    // once initialized, this function takes a mutable reference to allow
    // using a state field used to make verifying children easier. I can
    // make the state field in the struct RefCell later.
    // TODO: think about making SC a genertic for the trait itself
    fn verify(&mut self, engine: &E) -> Result<(), VerificationError>;
    // TODO: maybe look into default implementations for these
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[derive(PartialEq, Clone, Debug)]
pub struct Commitment<const LEN: usize> {
    commit: [u32; LEN],
}

impl<const LEN: usize> Default for Commitment<LEN> {
    fn default() -> Self {
        Self { commit: [0; LEN] }
    }
}

impl<const LEN: usize> Commitment<LEN> {
    pub fn from_slice(slice: &[u32]) -> Self {
        Self {
            commit: slice.try_into().unwrap(),
        }
    }

    pub fn flatten(&self) -> Vec<u32> {
        self.commit.to_vec()
    }
}
