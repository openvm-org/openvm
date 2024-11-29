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
    intrinsics::{
        ecc::{
            fp12::Fp12MulChip,
            fp2::{Fp2AddSubChip, Fp2MulDivChip},
            pairing::{
                EcLineMul013By013Chip, EcLineMul023By023Chip, EcLineMulBy01234Chip,
                EcLineMulBy02345Chip, EvaluateLineChip, MillerDoubleAndAddStepChip,
                MillerDoubleStepChip,
            },
            weierstrass::{EcAddNeChip, EcDoubleChip},
        },
        modular::{ModularAddSubChip, ModularIsEqualChip, ModularMulDivChip},
    },
    rv32im::*,
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
    // Rv32 (for standard 32-bit integers):
    BaseAluRv32(Rc<RefCell<Rv32BaseAluChip<F>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    LoadSignExtendRv32(Rc<RefCell<Rv32LoadSignExtendChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    // Intrinsics:
    HintStoreRv32(Rc<RefCell<Rv32HintStoreChip<F>>>),
    // Modular arithmetic:
    // 32-bytes or 48-bytes modulus.
    ModularAddSubRv32_1x32(Rc<RefCell<ModularAddSubChip<F, 1, 32>>>),
    ModularMulDivRv32_1x32(Rc<RefCell<ModularMulDivChip<F, 1, 32>>>),
    ModularAddSubRv32_3x16(Rc<RefCell<ModularAddSubChip<F, 3, 16>>>),
    ModularMulDivRv32_3x16(Rc<RefCell<ModularMulDivChip<F, 3, 16>>>),
    ModularIsEqualRv32_1x32(Rc<RefCell<ModularIsEqualChip<F, 1, 32, 32>>>),
    ModularIsEqualRv32_3x16(Rc<RefCell<ModularIsEqualChip<F, 3, 16, 48>>>),
    EcAddNeRv32_2x32(Rc<RefCell<EcAddNeChip<F, 2, 32>>>),
    EcDoubleRv32_2x32(Rc<RefCell<EcDoubleChip<F, 2, 32>>>),
    EcAddNeRv32_6x16(Rc<RefCell<EcAddNeChip<F, 6, 16>>>),
    EcDoubleRv32_6x16(Rc<RefCell<EcDoubleChip<F, 6, 16>>>),
    // Pairing:
    // Fp2 for 32-bytes or 48-bytes prime.
    Fp2AddSubRv32_32(Rc<RefCell<Fp2AddSubChip<F, 2, 32>>>),
    Fp2AddSubRv32_48(Rc<RefCell<Fp2AddSubChip<F, 6, 16>>>),
    Fp2MulDivRv32_32(Rc<RefCell<Fp2MulDivChip<F, 2, 32>>>),
    Fp2MulDivRv32_48(Rc<RefCell<Fp2MulDivChip<F, 6, 16>>>),
    // Fp12 for 32-bytes or 48-bytes prime.
    Fp12MulRv32_32(Rc<RefCell<Fp12MulChip<F, 12, 32>>>),
    Fp12MulRv32_48(Rc<RefCell<Fp12MulChip<F, 36, 16>>>),
    /// Only for BN254 for now
    EcLineMul013By013(Rc<RefCell<EcLineMul013By013Chip<F, 4, 10, 32>>>),
    /// Only for BN254 for now
    EcLineMulBy01234(Rc<RefCell<EcLineMulBy01234Chip<F, 12, 10, 12, 32>>>),
    /// Only for BLS12-381 for now
    EcLineMul023By023(Rc<RefCell<EcLineMul023By023Chip<F, 12, 30, 16>>>),
    /// Only for BLS12-381 for now
    EcLineMulBy02345(Rc<RefCell<EcLineMulBy02345Chip<F, 36, 30, 36, 16>>>),
    MillerDoubleStepRv32_32(Rc<RefCell<MillerDoubleStepChip<F, 4, 8, 32>>>),
    MillerDoubleStepRv32_48(Rc<RefCell<MillerDoubleStepChip<F, 12, 24, 16>>>),
    MillerDoubleAndAddStepRv32_32(Rc<RefCell<MillerDoubleAndAddStepChip<F, 4, 12, 32>>>),
    MillerDoubleAndAddStepRv32_48(Rc<RefCell<MillerDoubleAndAddStepChip<F, 12, 36, 16>>>),
    EvaluateLineRv32_32(Rc<RefCell<EvaluateLineChip<F, 4, 2, 4, 32>>>),
    EvaluateLineRv32_48(Rc<RefCell<EvaluateLineChip<F, 12, 6, 12, 16>>>),
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
