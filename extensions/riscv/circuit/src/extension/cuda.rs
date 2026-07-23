use std::sync::Arc;

use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension,
    },
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU},
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
    },
    openvm_circuit::system::cuda::poseidon2::Poseidon2PeripheryChipGPU,
    openvm_circuit_primitives::{AnyChip, Chip},
    openvm_cuda_backend::base::DeviceMatrix,
    openvm_instructions::{LocalOpcode, SystemOpcode},
    openvm_riscv_transpiler::{
        BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
        BranchLessThanOpcode, LessThanImmOpcode, LessThanOpcode, Rv64AuipcOpcode, Rv64JalLuiOpcode,
        Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode,
        ShiftWOpcode,
    },
    openvm_stark_backend::prover::AirProvingContext,
};

use crate::{
    Rv64AddIAir, Rv64AddIChipGpu, Rv64AddIWAir, Rv64AddIWChipGpu, Rv64AddSubAir, Rv64AddSubChipGpu,
    Rv64AddSubWAir, Rv64AddSubWChipGpu, Rv64AuipcAir, Rv64AuipcChipGpu, Rv64BitwiseLogicAir,
    Rv64BitwiseLogicChipGpu, Rv64BitwiseLogicImmAir, Rv64BitwiseLogicImmChipGpu,
    Rv64BranchEqualAir, Rv64BranchEqualChipGpu, Rv64BranchLessThanAir, Rv64BranchLessThanChipGpu,
    Rv64DivRemAir, Rv64DivRemChipGpu, Rv64DivRemWAir, Rv64DivRemWChipGpu, Rv64HintStoreAir,
    Rv64HintStoreChipGpu, Rv64I, Rv64Io, Rv64JalLuiAir, Rv64JalLuiChipGpu, Rv64JalrAir,
    Rv64JalrChipGpu, Rv64LessThanAir, Rv64LessThanChipGpu, Rv64LessThanImmAir,
    Rv64LessThanImmChipGpu, Rv64LoadByteAir, Rv64LoadByteChipGpu, Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChipGpu, Rv64LoadHalfwordAir, Rv64LoadHalfwordChipGpu,
    Rv64LoadSignExtendByteAir, Rv64LoadSignExtendByteChipGpu, Rv64LoadSignExtendHalfwordAir,
    Rv64LoadSignExtendHalfwordChipGpu, Rv64LoadSignExtendWordAir, Rv64LoadSignExtendWordChipGpu,
    Rv64LoadWordAir, Rv64LoadWordChipGpu, Rv64M, Rv64MulHAir, Rv64MulHChipGpu, Rv64MulWAir,
    Rv64MulWChipGpu, Rv64MultiplicationAir, Rv64MultiplicationChipGpu, Rv64ShiftLogicalAir,
    Rv64ShiftLogicalChipGpu, Rv64ShiftLogicalImmAir, Rv64ShiftLogicalImmChipGpu,
    Rv64ShiftRightArithmeticAir, Rv64ShiftRightArithmeticChipGpu, Rv64ShiftRightArithmeticImmAir,
    Rv64ShiftRightArithmeticImmChipGpu, Rv64ShiftWLogicalAir, Rv64ShiftWLogicalChipGpu,
    Rv64ShiftWLogicalImmAir, Rv64ShiftWLogicalImmChipGpu, Rv64ShiftWRightArithmeticAir,
    Rv64ShiftWRightArithmeticChipGpu, Rv64ShiftWRightArithmeticImmAir,
    Rv64ShiftWRightArithmeticImmChipGpu, Rv64StoreByteAir, Rv64StoreByteChipGpu,
    Rv64StoreDoublewordAir, Rv64StoreDoublewordChipGpu, Rv64StoreHalfwordAir,
    Rv64StoreHalfwordChipGpu, Rv64StoreWordAir, Rv64StoreWordChipGpu,
};

pub struct Rv64ImGpuProverExt;

/// Segment-wide RV64I GPU trace generation from an immutable RVR transcript.
///
/// Construction rejects an executed opcode unless its trace kernel is present
/// below (or it is the system-owned `TERMINATE`). Each supported opcode remains
/// pending until the VM's reverse inventory walk reaches its concrete chip.
/// This makes a missing/mismatched chip fail closed instead of silently
/// producing a dummy trace.
#[cfg(feature = "rvr")]
pub struct Rv64IRvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    pending_opcodes: std::collections::BTreeSet<u32>,
}

#[cfg(feature = "rvr")]
impl<'a> Rv64IRvrGpuTracegen<'a> {
    pub fn new(
        program: &'a GpuRvrProgram,
        transcript: &'a GpuRvrTranscript,
        replay_plan: &'a GpuRvrReplayPlan,
    ) -> Result<Self, GpuRvrInputError> {
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let pending_opcodes = replay_plan
            .executed_opcodes()
            .filter(|&opcode| opcode != terminate)
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(&opcode) = pending_opcodes
            .iter()
            .find(|&&opcode| !Self::supports_opcode(opcode))
        {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "RV64I RVR GPU tracegen does not support executed opcode {opcode:#x}"
            )));
        }
        Ok(Self {
            program,
            transcript,
            replay_plan,
            pending_opcodes,
        })
    }

    fn supports_opcode(opcode: u32) -> bool {
        [
            BaseAluOpcode::ADD.global_opcode(),
            BaseAluOpcode::SUB.global_opcode(),
            BaseAluOpcode::XOR.global_opcode(),
            BaseAluOpcode::OR.global_opcode(),
            BaseAluOpcode::AND.global_opcode(),
            BaseAluWOpcode::ADDW.global_opcode(),
            BaseAluWOpcode::SUBW.global_opcode(),
            LessThanOpcode::SLT.global_opcode(),
            LessThanOpcode::SLTU.global_opcode(),
            ShiftOpcode::SLL.global_opcode(),
            ShiftOpcode::SRL.global_opcode(),
            ShiftOpcode::SRA.global_opcode(),
            ShiftWOpcode::SLLW.global_opcode(),
            ShiftWOpcode::SRLW.global_opcode(),
            ShiftWOpcode::SRAW.global_opcode(),
            BaseAluWImmOpcode::ADDIW.global_opcode(),
            ShiftWImmOpcode::SLLIW.global_opcode(),
            ShiftWImmOpcode::SRLIW.global_opcode(),
            ShiftWImmOpcode::SRAIW.global_opcode(),
            BranchEqualOpcode::BEQ.global_opcode(),
            BranchEqualOpcode::BNE.global_opcode(),
            BranchLessThanOpcode::BLT.global_opcode(),
            BranchLessThanOpcode::BLTU.global_opcode(),
            BranchLessThanOpcode::BGE.global_opcode(),
            BranchLessThanOpcode::BGEU.global_opcode(),
            Rv64JalLuiOpcode::JAL.global_opcode(),
            Rv64JalLuiOpcode::LUI.global_opcode(),
            Rv64JalrOpcode::JALR.global_opcode(),
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            BaseAluImmOpcode::ADDI.global_opcode(),
            ShiftImmOpcode::SLLI.global_opcode(),
            ShiftImmOpcode::SRLI.global_opcode(),
            ShiftImmOpcode::SRAI.global_opcode(),
            LessThanImmOpcode::SLTI.global_opcode(),
            LessThanImmOpcode::SLTIU.global_opcode(),
            BaseAluImmOpcode::XORI.global_opcode(),
            BaseAluImmOpcode::ORI.global_opcode(),
            BaseAluImmOpcode::ANDI.global_opcode(),
            Rv64LoadStoreOpcode::LOADB.global_opcode(),
            Rv64LoadStoreOpcode::LOADBU.global_opcode(),
            Rv64LoadStoreOpcode::LOADH.global_opcode(),
            Rv64LoadStoreOpcode::LOADHU.global_opcode(),
            Rv64LoadStoreOpcode::LOADW.global_opcode(),
            Rv64LoadStoreOpcode::LOADWU.global_opcode(),
            Rv64LoadStoreOpcode::LOADD.global_opcode(),
            // The concrete replay kernel accepts the RV64I main-memory shape.
            // RV64IO public-values stores fail closed in instruction validation.
            Rv64LoadStoreOpcode::STOREB.global_opcode(),
        ]
        .into_iter()
        .any(|candidate| candidate.as_usize() as u32 == opcode)
    }

    fn mark_generated(&mut self, opcodes: impl IntoIterator<Item = u32>) {
        for opcode in opcodes {
            self.pending_opcodes.remove(&opcode);
        }
    }

    fn opcode(opcode: impl LocalOpcode) -> u32 {
        opcode.global_opcode().as_usize() as u32
    }

    /// Generates one extension AIR in the VM inventory's normal reverse order.
    ///
    /// Replay producers update their shared lookup histograms. Periphery chips
    /// are then generated from those histograms through their ordinary
    /// record-independent path. Every other chip is known to be unexecuted by
    /// the constructor coverage check, so it receives a dummy trace without
    /// touching a `RecordArena`.
    pub fn generate_for_chip(
        &mut self,
        _insertion_idx: usize,
        chip: &dyn AnyChip<DenseRecordArena, GpuBackend>,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        macro_rules! replay_chip {
            ($chip_ty:ty, [$($opcode:expr),+ $(,)?]) => {
                if let Some(chip) = chip.as_any().downcast_ref::<$chip_ty>() {
                    self.mark_generated([$(
                        Self::opcode($opcode)
                    ),+]);
                    return chip.generate_proving_ctx_from_rvr(
                        self.program,
                        self.transcript,
                        self.replay_plan,
                    );
                }
            };
        }

        replay_chip!(Rv64AddSubChipGpu, [BaseAluOpcode::ADD, BaseAluOpcode::SUB]);
        replay_chip!(
            Rv64BitwiseLogicChipGpu,
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND,]
        );
        replay_chip!(
            Rv64AddSubWChipGpu,
            [BaseAluWOpcode::ADDW, BaseAluWOpcode::SUBW]
        );
        replay_chip!(
            Rv64LessThanChipGpu,
            [LessThanOpcode::SLT, LessThanOpcode::SLTU]
        );
        replay_chip!(
            Rv64ShiftLogicalChipGpu,
            [ShiftOpcode::SLL, ShiftOpcode::SRL]
        );
        replay_chip!(Rv64ShiftRightArithmeticChipGpu, [ShiftOpcode::SRA]);
        replay_chip!(
            Rv64ShiftWLogicalChipGpu,
            [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW]
        );
        replay_chip!(Rv64ShiftWRightArithmeticChipGpu, [ShiftWOpcode::SRAW]);
        replay_chip!(Rv64AddIWChipGpu, [BaseAluWImmOpcode::ADDIW]);
        replay_chip!(
            Rv64ShiftWLogicalImmChipGpu,
            [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW]
        );
        replay_chip!(
            Rv64ShiftWRightArithmeticImmChipGpu,
            [ShiftWImmOpcode::SRAIW]
        );
        replay_chip!(
            Rv64BranchEqualChipGpu,
            [BranchEqualOpcode::BEQ, BranchEqualOpcode::BNE]
        );
        replay_chip!(
            Rv64BranchLessThanChipGpu,
            [
                BranchLessThanOpcode::BLT,
                BranchLessThanOpcode::BLTU,
                BranchLessThanOpcode::BGE,
                BranchLessThanOpcode::BGEU,
            ]
        );
        replay_chip!(
            Rv64JalLuiChipGpu,
            [Rv64JalLuiOpcode::JAL, Rv64JalLuiOpcode::LUI]
        );
        replay_chip!(Rv64JalrChipGpu, [Rv64JalrOpcode::JALR]);
        replay_chip!(Rv64AuipcChipGpu, [Rv64AuipcOpcode::AUIPC]);
        replay_chip!(Rv64LoadSignExtendByteChipGpu, [Rv64LoadStoreOpcode::LOADB]);
        replay_chip!(Rv64LoadByteChipGpu, [Rv64LoadStoreOpcode::LOADBU]);
        replay_chip!(
            Rv64LoadSignExtendHalfwordChipGpu,
            [Rv64LoadStoreOpcode::LOADH]
        );
        replay_chip!(Rv64LoadHalfwordChipGpu, [Rv64LoadStoreOpcode::LOADHU]);
        replay_chip!(Rv64LoadSignExtendWordChipGpu, [Rv64LoadStoreOpcode::LOADW]);
        replay_chip!(Rv64LoadWordChipGpu, [Rv64LoadStoreOpcode::LOADWU]);
        replay_chip!(Rv64LoadDoublewordChipGpu, [Rv64LoadStoreOpcode::LOADD]);
        replay_chip!(Rv64StoreByteChipGpu, [Rv64LoadStoreOpcode::STOREB]);
        replay_chip!(Rv64AddIChipGpu, [BaseAluImmOpcode::ADDI]);
        replay_chip!(
            Rv64ShiftLogicalImmChipGpu,
            [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI]
        );
        replay_chip!(Rv64ShiftRightArithmeticImmChipGpu, [ShiftImmOpcode::SRAI]);
        replay_chip!(
            Rv64LessThanImmChipGpu,
            [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU]
        );
        replay_chip!(
            Rv64BitwiseLogicImmChipGpu,
            [
                BaseAluImmOpcode::XORI,
                BaseAluImmOpcode::ORI,
                BaseAluImmOpcode::ANDI,
            ]
        );

        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Arc<VariableRangeCheckerChipGPU>>()
        {
            return Ok(
                <Arc<VariableRangeCheckerChipGPU> as Chip<(), GpuBackend>>::generate_proving_ctx(
                    chip,
                    (),
                ),
            );
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Arc<BitwiseOperationLookupChipGPU<8>>>()
        {
            return Ok(<Arc<BitwiseOperationLookupChipGPU<8>> as Chip<
                (),
                GpuBackend,
            >>::generate_proving_ctx(chip, ()));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Arc<Poseidon2PeripheryChipGPU>>()
        {
            return Ok(
                <Arc<Poseidon2PeripheryChipGPU> as Chip<(), GpuBackend>>::generate_proving_ctx(
                    chip,
                    (),
                ),
            );
        }

        Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()))
    }

    /// Completes one segment after every extension AIR context has been made.
    ///
    /// This performs the one post-replay synchronized read of the sticky GPU
    /// error word. Call it immediately before handing the context to the proof
    /// engine.
    pub fn finish(self) -> Result<(), GpuRvrInputError> {
        if !self.pending_opcodes.is_empty() {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "RV64I RVR GPU tracegen did not visit chips for executed opcodes {:?}",
                self.pending_opcodes
            )));
        }
        let error = self.transcript.error_code()?;
        if error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "RV64I RVR GPU tracegen rejected transcript with code {error}"
            )));
        }
        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64I> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64AddSubAir>()?;
        let add_sub = Rv64AddSubChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<Rv64BitwiseLogicAir>()?;
        let bitwise_logic = Rv64BitwiseLogicChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(bitwise_logic);

        inventory.next_air::<Rv64AddSubWAir>()?;
        let add_sub_w = Rv64AddSubWChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(add_sub_w);

        inventory.next_air::<Rv64LessThanAir>()?;
        let lt = Rv64LessThanChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64ShiftLogicalAir>()?;
        let shift_logical = Rv64ShiftLogicalChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<Rv64ShiftRightArithmeticAir>()?;
        let shift_right_arithmetic =
            Rv64ShiftRightArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_right_arithmetic);

        inventory.next_air::<Rv64ShiftWLogicalAir>()?;
        let shift_w_logical =
            Rv64ShiftWLogicalChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_logical);

        inventory.next_air::<Rv64ShiftWRightArithmeticAir>()?;
        let shift_w_right_arithmetic =
            Rv64ShiftWRightArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_right_arithmetic);

        inventory.next_air::<Rv64AddIWAir>()?;
        let addi_w = Rv64AddIWChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(addi_w);

        inventory.next_air::<Rv64ShiftWLogicalImmAir>()?;
        let shift_w_logical_imm =
            Rv64ShiftWLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_logical_imm);

        inventory.next_air::<Rv64ShiftWRightArithmeticImmAir>()?;
        let shift_w_right_arithmetic_imm =
            Rv64ShiftWRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_right_arithmetic_imm);

        inventory.next_air::<Rv64LoadSignExtendByteAir>()?;
        let load_sign_extend_byte = Rv64LoadSignExtendByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_byte);

        inventory.next_air::<Rv64LoadByteAir>()?;
        let load_byte = Rv64LoadByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_byte);

        inventory.next_air::<Rv64StoreByteAir>()?;
        let store_byte = Rv64StoreByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_byte);

        inventory.next_air::<Rv64LoadSignExtendHalfwordAir>()?;
        let load_sign_extend_halfword = Rv64LoadSignExtendHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_halfword);

        inventory.next_air::<Rv64LoadHalfwordAir>()?;
        let load_halfword = Rv64LoadHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_halfword);

        inventory.next_air::<Rv64StoreHalfwordAir>()?;
        let store_halfword = Rv64StoreHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_halfword);

        inventory.next_air::<Rv64LoadSignExtendWordAir>()?;
        let load_sign_extend_word = Rv64LoadSignExtendWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_word);

        inventory.next_air::<Rv64LoadWordAir>()?;
        let load_word = Rv64LoadWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_word);

        inventory.next_air::<Rv64StoreWordAir>()?;
        let store_word = Rv64StoreWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_word);

        inventory.next_air::<Rv64LoadDoublewordAir>()?;
        let load_doubleword = Rv64LoadDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_doubleword);

        inventory.next_air::<Rv64StoreDoublewordAir>()?;
        let store_doubleword = Rv64StoreDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_doubleword);

        inventory.next_air::<Rv64BranchEqualAir>()?;
        let beq = Rv64BranchEqualChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThanAir>()?;
        let blt = Rv64BranchLessThanChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64JalLuiAir>()?;
        let jal_lui = Rv64JalLuiChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv64JalrAir>()?;
        let jalr = Rv64JalrChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv64AuipcAir>()?;
        let auipc = Rv64AuipcChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(auipc);

        inventory.next_air::<Rv64AddIAir>()?;
        let addi = Rv64AddIChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(addi);

        inventory.next_air::<Rv64ShiftLogicalImmAir>()?;
        let shift_logical_imm =
            Rv64ShiftLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_logical_imm);

        inventory.next_air::<Rv64ShiftRightArithmeticImmAir>()?;
        let shift_right_arithmetic_imm =
            Rv64ShiftRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_right_arithmetic_imm);

        inventory.next_air::<Rv64LessThanImmAir>()?;
        let lt_imm = Rv64LessThanImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(lt_imm);

        inventory.next_air::<Rv64BitwiseLogicImmAir>()?;
        let bitwise_logic_imm = Rv64BitwiseLogicImmChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(bitwise_logic_imm);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64M> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                    range_checker.device_ctx.clone(),
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64MultiplicationAir>()?;
        let mult = Rv64MultiplicationChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64MulWAir>()?;
        let mul_w = Rv64MulWChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_w);

        inventory.next_air::<Rv64MulHAir>()?;
        let mul_h = Rv64MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv64DivRemAir>()?;
        let div_rem = Rv64DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(div_rem);

        inventory.next_air::<Rv64DivRemWAir>()?;
        let divrem_w = Rv64DivRemWChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(divrem_w);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64Io>
    for Rv64ImGpuProverExt
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);

        inventory.next_air::<Rv64HintStoreAir>()?;
        let hint_store =
            Rv64HintStoreChipGpu::new(range_checker.clone(), byte_ptr_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests;
