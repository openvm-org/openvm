use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip, range_tuple::RangeTupleCheckerChip,
};
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::AirProofInput,
    Chip,
};
use axvm_instructions::instruction::Instruction;
use derive_more::From;
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;

use crate::{
    arch::ExecutionState,
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
        hashes::{keccak256::KeccakVmChip, poseidon2::Poseidon2Chip},
        int256::{
            Rv32BaseAlu256Chip, Rv32BranchEqual256Chip, Rv32BranchLessThan256Chip,
            Rv32LessThan256Chip, Rv32Multiplication256Chip, Rv32Shift256Chip,
        },
        modular::{ModularAddSubChip, ModularIsEqualChip, ModularMulDivChip},
    },
    kernels::{
        branch_eq::KernelBranchEqChip, castf::CastFChip, field_arithmetic::FieldArithmeticChip,
        field_extension::FieldExtensionChip, fri::FriReducedOpeningChip, jal::KernelJalChip,
        loadstore::KernelLoadStoreChip, public_values::PublicValuesChip,
    },
    rv32im::*,
    system::{phantom::PhantomChip, program::ExecutionError},
};

pub trait InstructionExecutor<F> {
    /// Runtime execution of the instruction, if the instruction is owned by the
    /// current instance. May internally store records of this call for later trace generation.
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError>;

    /// For display purposes. From absolute opcode as `usize`, return the string name of the opcode
    /// if it is a supported opcode by the present executor.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

impl<F, C: InstructionExecutor<F>> InstructionExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        prev_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        self.borrow_mut().execute(instruction, prev_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.borrow().get_opcode_name(opcode)
    }
}

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(EnumDiscriminants, ChipUsageGetter, Chip, InstructionExecutor, From)]
#[strum_discriminants(derive(Serialize, Deserialize, Ord, PartialOrd))]
#[strum_discriminants(name(ExecutorName))]
pub enum AxVmExecutor<F: PrimeField32> {
    Phantom(Rc<RefCell<PhantomChip<F>>>),
    // Native kernel:
    LoadStore(Rc<RefCell<KernelLoadStoreChip<F, 1>>>),
    BranchEqual(Rc<RefCell<KernelBranchEqChip<F>>>),
    Jal(Rc<RefCell<KernelJalChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionChip<F>>>),
    PublicValues(Rc<RefCell<PublicValuesChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    FriReducedOpening(Rc<RefCell<FriReducedOpeningChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
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
    Keccak256Rv32(Rc<RefCell<KeccakVmChip<F>>>),
    // 256Rv32 (for 256-bit integers):
    BaseAlu256Rv32(Rc<RefCell<Rv32BaseAlu256Chip<F>>>),
    Shift256Rv32(Rc<RefCell<Rv32Shift256Chip<F>>>),
    LessThan256Rv32(Rc<RefCell<Rv32LessThan256Chip<F>>>),
    BranchEqual256Rv32(Rc<RefCell<Rv32BranchEqual256Chip<F>>>),
    BranchLessThan256Rv32(Rc<RefCell<Rv32BranchLessThan256Chip<F>>>),
    Multiplication256Rv32(Rc<RefCell<Rv32Multiplication256Chip<F>>>),
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
        let height = height.next_power_of_two();
        let mut proof_input = self.generate_air_proof_input();
        let main = proof_input.raw.common_main.as_mut().unwrap();
        assert!(
            height >= main.height(),
            "Overridden height must be greater than or equal to the used height"
        );
        // Assumption: an all-0 row is a valid dummy row for all chips.
        main.pad_to_height(height, F::ZERO);
        proof_input
    }
}

// NOTE[yj]: Debug. delete this.
impl<F: PrimeField32> AxVmExecutor<F> {
    pub fn executor_name(&self) -> &'static str {
        match self {
            AxVmExecutor::Phantom(_) => "Phantom",
            AxVmExecutor::LoadStore(_) => "LoadStore",
            AxVmExecutor::BranchEqual(_) => "BranchEqual",
            AxVmExecutor::Jal(_) => "Jal",
            AxVmExecutor::FieldArithmetic(_) => "FieldArithmetic",
            AxVmExecutor::FieldExtension(_) => "FieldExtension",
            AxVmExecutor::PublicValues(_) => "PublicValues",
            AxVmExecutor::Poseidon2(_) => "Poseidon2",
            AxVmExecutor::FriReducedOpening(_) => "FriReducedOpening",
            AxVmExecutor::CastF(_) => "CastF",
            AxVmExecutor::BaseAluRv32(_) => "BaseAluRv32",
            AxVmExecutor::LessThanRv32(_) => "LessThanRv32",
            AxVmExecutor::ShiftRv32(_) => "ShiftRv32",
            AxVmExecutor::LoadStoreRv32(_) => "LoadStoreRv32",
            AxVmExecutor::LoadSignExtendRv32(_) => "LoadSignExtendRv32",
            AxVmExecutor::BranchEqualRv32(_) => "BranchEqualRv32",
            AxVmExecutor::BranchLessThanRv32(_) => "BranchLessThanRv32",
            AxVmExecutor::JalLuiRv32(_) => "JalLuiRv32",
            AxVmExecutor::JalrRv32(_) => "JalrRv32",
            AxVmExecutor::AuipcRv32(_) => "AuipcRv32",
            AxVmExecutor::MultiplicationRv32(_) => "MultiplicationRv32",
            AxVmExecutor::MultiplicationHighRv32(_) => "MultiplicationHighRv32",
            AxVmExecutor::DivRemRv32(_) => "DivRemRv32",
            AxVmExecutor::HintStoreRv32(_) => "HintStoreRv32",
            AxVmExecutor::Keccak256Rv32(_) => "Keccak256Rv32",
            AxVmExecutor::BaseAlu256Rv32(_) => "BaseAlu256Rv32",
            AxVmExecutor::Shift256Rv32(_) => "Shift256Rv32",
            AxVmExecutor::LessThan256Rv32(_) => "LessThan256Rv32",
            AxVmExecutor::BranchEqual256Rv32(_) => "BranchEqual256Rv32",
            AxVmExecutor::BranchLessThan256Rv32(_) => "BranchLessThan256Rv32",
            AxVmExecutor::Multiplication256Rv32(_) => "Multiplication256Rv32",
            AxVmExecutor::ModularAddSubRv32_1x32(_) => "ModularAddSubRv32_1x32",
            AxVmExecutor::ModularMulDivRv32_1x32(_) => "ModularMulDivRv32_1x32",
            AxVmExecutor::ModularAddSubRv32_3x16(_) => "ModularAddSubRv32_3x16",
            AxVmExecutor::ModularMulDivRv32_3x16(_) => "ModularMulDivRv32_3x16",
            AxVmExecutor::EcAddNeRv32_2x32(_) => "EcAddNeRv32_2x32",
            AxVmExecutor::EcDoubleRv32_2x32(_) => "EcDoubleRv32_2x32",
            AxVmExecutor::EcAddNeRv32_6x16(_) => "EcAddNeRv32_6x16",
            AxVmExecutor::EcDoubleRv32_6x16(_) => "EcDoubleRv32_6x16",
            AxVmExecutor::Fp2AddSubRv32_32(_) => "Fp2AddSubRv32_32",
            AxVmExecutor::Fp2AddSubRv32_48(_) => "Fp2AddSubRv32_48",
            AxVmExecutor::Fp2MulDivRv32_32(_) => "Fp2MulDivRv32_32",
            AxVmExecutor::Fp2MulDivRv32_48(_) => "Fp2MulDivRv32_48",
            AxVmExecutor::Fp12MulRv32_32(_) => "Fp12MulRv32_32",
            AxVmExecutor::Fp12MulRv32_48(_) => "Fp12MulRv32_48",
            AxVmExecutor::EcLineMul013By013(_) => "EcLineMul013By013",
            AxVmExecutor::EcLineMulBy01234(_) => "EcLineMulBy01234",
            AxVmExecutor::EcLineMul023By023(_) => "EcLineMul023By023",
            AxVmExecutor::EcLineMulBy02345(_) => "EcLineMulBy02345",
            AxVmExecutor::MillerDoubleStepRv32_32(_) => "MillerDoubleStepRv32_32",
            AxVmExecutor::MillerDoubleStepRv32_48(_) => "MillerDoubleStepRv32_48",
            AxVmExecutor::MillerDoubleAndAddStepRv32_32(_) => "MillerDoubleAndAddStepRv32_32",
            AxVmExecutor::MillerDoubleAndAddStepRv32_48(_) => "MillerDoubleAndAddStepRv32_48",
            AxVmExecutor::EvaluateLineRv32_32(_) => "EvaluateLineRv32_32",
            AxVmExecutor::EvaluateLineRv32_48(_) => "EvaluateLineRv32_48",
            AxVmExecutor::ModularIsEqualRv32_1x32(_) => "ModularIsEqualRv32_1x32",
            AxVmExecutor::ModularIsEqualRv32_3x16(_) => "ModularIsEqualRv32_3x16",
        }
    }
}
