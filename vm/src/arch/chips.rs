use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip, range_tuple::RangeTupleCheckerChip,
};
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::AirProofInput,
};
use derive_more::From;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;

use crate::{
    derive::InstructionExecutor,
    system::{phantom::PhantomChip, poseidon2::Poseidon2Chip, public_values::PublicValuesChip},
};

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(EnumDiscriminants, ChipUsageGetter, Chip, InstructionExecutor, From)]
#[strum_discriminants(derive(Serialize, Deserialize, Ord, PartialOrd))]
#[strum_discriminants(name(ExecutorName))]
pub enum AxVmExecutor<F: PrimeField32> {
    Phantom(Rc<RefCell<PhantomChip<F>>>),
    // Native kernel:
    PublicValues(Rc<RefCell<PublicValuesChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
}

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(From, ChipUsageGetter, Chip)]
pub enum AxVmChip<F: PrimeField32> {
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // Instruction Executors
    Executor(AxVmExecutor<F>),
}

impl<F: PrimeField32> AxVmExecutor<F> {
    /// Generates an AIR proof input of the chip with the given height.
    pub fn generate_air_proof_input_with_height<SC: StarkGenericConfig>(
        self,
        height: usize,
    ) -> AirProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        super::generate_air_proof_input(self, Some(height))
    }
}
