use std::{cell::RefCell, rc::Rc, sync::Arc};

use p3_uni_stark::StarkGenericConfig;

use crate::{prover::types::AirProofInput, rap::AnyRap};

/// A chip is a stateful struct that stores the state necessary to
/// generate the trace of an AIR. This trait is for proving purposes
/// and has a generic [StarkGenericConfig] since it needs to know the STARK config.
pub trait Chip<SC: StarkGenericConfig> {
    fn air(&self) -> Arc<dyn AnyRap<SC>>;
    /// Generate all necessary input for proving a single AIR.
    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        // TEMPORARY[jpw]: make it easier to implement this trait in transition
        todo!();
    }
    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, AirProofInput<SC>) {
        (air_id, self.generate_air_proof_input())
    }
}

impl<SC: StarkGenericConfig, C: Chip<SC>> Chip<SC> for Rc<RefCell<C>> {
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        self.borrow().air()
    }
    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        self.borrow().generate_air_proof_input()
    }
    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, AirProofInput<SC>) {
        self.borrow().generate_air_proof_input_with_id(air_id)
    }
}

impl<SC: StarkGenericConfig, C: Chip<SC>> Chip<SC> for Arc<C> {
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        self.as_ref().air()
    }
    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        self.as_ref().generate_air_proof_input()
    }
    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, AirProofInput<SC>) {
        self.as_ref().generate_air_proof_input_with_id(air_id)
    }
}
