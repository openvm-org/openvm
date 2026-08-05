use std::{any::Any, sync::Arc};

#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::PreflightExecution;
use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        prepare_gpu_postflight, to_byte_ptr_bits, ChipInventory, ChipInventoryError,
        GenerationError, PostflightTracegen, PreflightOutput, VirtualMachine, VmBuilder,
        VmProverExtension,
    },
    system::cuda::{
        extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
        phantom::PhantomChipGPU,
        poseidon2::Poseidon2PeripheryChipGPU,
        SystemChipInventoryGPU,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU},
    var_range::VariableRangeCheckerChipGPU,
    Chip,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_instructions::{program::Program, LocalOpcode, SystemOpcode};
use openvm_riscv_transpiler::{
    AuipcOpcode, BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode,
    BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, HintStoreOpcode,
    JalLuiOpcode, JalrOpcode, LessThanImmOpcode, LessThanOpcode, LoadStoreOpcode, MulHOpcode,
    MulOpcode, MulWOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
};
use openvm_stark_backend::prover::{AirProvingContext, ProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};

#[cfg(feature = "rvr")]
use crate::preflight::PreflightReplayProgram;
use crate::{
    AddIAir, AddIChipGpu, AddIWAir, AddIWChipGpu, AddSubAir, AddSubChipGpu, AddSubWAir,
    AddSubWChipGpu, AuipcAir, AuipcChipGpu, BitwiseLogicAir, BitwiseLogicChipGpu,
    BitwiseLogicImmAir, BitwiseLogicImmChipGpu, BranchEqualAir, BranchEqualChipGpu,
    BranchLessThanAir, BranchLessThanChipGpu, DivRemAir, DivRemChipGpu, DivRemWAir, DivRemWChipGpu,
    HintStoreAir, HintStoreChipGpu, JalLuiAir, JalLuiChipGpu, JalrAir, JalrChipGpu, LessThanAir,
    LessThanChipGpu, LessThanImmAir, LessThanImmChipGpu, LoadByteAir, LoadByteChipGpu,
    LoadDoublewordAir, LoadDoublewordChipGpu, LoadHalfwordAir, LoadHalfwordChipGpu,
    LoadSignExtendByteAir, LoadSignExtendByteChipGpu, LoadSignExtendHalfwordAir,
    LoadSignExtendHalfwordChipGpu, LoadSignExtendWordAir, LoadSignExtendWordChipGpu, LoadWordAir,
    LoadWordChipGpu, MulHAir, MulHChipGpu, MulWAir, MulWChipGpu, MultiplicationAir,
    MultiplicationChipGpu, Rv64I, Rv64Io, Rv64M, ShiftLogicalAir, ShiftLogicalChipGpu,
    ShiftLogicalImmAir, ShiftLogicalImmChipGpu, ShiftRightArithmeticAir,
    ShiftRightArithmeticChipGpu, ShiftRightArithmeticImmAir, ShiftRightArithmeticImmChipGpu,
    ShiftWLogicalAir, ShiftWLogicalChipGpu, ShiftWLogicalImmAir, ShiftWLogicalImmChipGpu,
    ShiftWRightArithmeticAir, ShiftWRightArithmeticChipGpu, ShiftWRightArithmeticImmAir,
    ShiftWRightArithmeticImmChipGpu, StoreByteAir, StoreByteChipGpu, StoreDoublewordAir,
    StoreDoublewordChipGpu, StoreHalfwordAir, StoreHalfwordChipGpu, StoreWordAir, StoreWordChipGpu,
};

include!(concat!(env!("OUT_DIR"), "/checkpoint_replay_opcodes.rs"));

pub struct Rv64ImGpuProverExt;

macro_rules! impl_postflight_tracegen {
    ($builder:ty) => {
        impl PostflightTracegen<GpuBabyBearPoseidon2Engine> for $builder {
            type Prepared = GpuPostflightProgram;

            fn prepare_postflight(
                vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
                program: &Program<F>,
            ) -> Result<Self::Prepared, GenerationError> {
                prepare_gpu_postflight(vm, program)
            }

            fn generate_proving_ctx(
                vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
                _host_program: &Program<F>,
                program: &Self::Prepared,
                output: &PreflightOutput,
            ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
                let (transcript, replay_plan) = vm
                    .postflight_history(program, output)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
                Rv64ImPreflightGpuTracegen::new(program, &transcript, &replay_plan)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                    .generate_proving_ctx(vm)
            }
        }
    };
}

impl_postflight_tracegen!(crate::Rv64IGpuBuilder);
impl_postflight_tracegen!(crate::Rv64ImGpuBuilder);

/// Segment-wide RV64I GPU trace generation from an immutable postflight transcript.
///
/// Construction rejects an executed opcode unless its trace kernel is present
/// below (or it is the record-free system `TERMINATE`). Each supported opcode
/// remains pending until the VM's reverse inventory walk reaches its concrete chip.
/// This makes a missing/mismatched chip fail closed instead of silently
/// producing a dummy trace.
pub struct Rv64ImPreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    pending_opcodes: std::collections::BTreeSet<u32>,
}

impl<'a> Rv64ImPreflightGpuTracegen<'a> {
    /// Checkpoint replay for RV64IM and phantom execution. Loads and stores
    /// first become unresolved block intents; the VM chronology pass resolves
    /// those intents before the ordinary transcript indexes and unchanged
    /// trace generators consume them.
    #[cfg(feature = "rvr")]
    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &PreflightReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        let context = vm.gpu_postflight_context(program.program())?;
        program.postflight(context, execution, num_insns)
    }

    pub fn new(
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
    ) -> Result<Self, GpuPostflightError> {
        Self::new_after_claiming_extension_opcodes(program, transcript, replay_plan, &[])
    }

    /// Constructs the RV64 producer after a concrete outer coordinator has
    /// claimed its extension opcodes.
    ///
    /// The caller remains responsible for visiting and finishing the claimed
    /// extension producers. Every remaining executed opcode is still checked
    /// against RV64's exact supported set here.
    #[doc(hidden)]
    pub fn new_after_claiming_extension_opcodes(
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
        extension_opcodes: &[u32],
    ) -> Result<Self, GpuPostflightError> {
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let extension_opcodes = extension_opcodes
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let pending_opcodes = replay_plan
            .executed_opcodes()
            .filter(|&opcode| opcode != terminate && !extension_opcodes.contains(&opcode))
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(&opcode) = pending_opcodes
            .iter()
            .find(|&&opcode| !Self::supports_opcode(opcode))
        {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "RV64IM preflight GPU tracegen does not support executed opcode {opcode:#x}"
            )));
        }
        Ok(Self {
            program,
            transcript,
            replay_plan,
            pending_opcodes,
        })
    }

    /// Returns whether the standard RV64/system replay path owns `opcode`.
    #[doc(hidden)]
    pub fn owns_opcode(opcode: u32) -> bool {
        Self::replay_opcodes().any(|candidate| candidate == opcode)
    }

    /// Opcodes implemented by the native RV64 replay kernel.
    pub(crate) fn replay_opcodes() -> impl Iterator<Item = u32> {
        REPLAY_OPCODES.iter().copied()
    }

    fn supports_opcode(opcode: u32) -> bool {
        Self::replay_opcodes().any(|candidate| candidate == opcode)
    }

    fn mark_generated(&mut self, opcodes: impl IntoIterator<Item = u32>) {
        for opcode in opcodes {
            self.pending_opcodes.remove(&opcode);
        }
    }

    fn opcode(opcode: impl LocalOpcode) -> u32 {
        opcode.global_opcode().as_usize() as u32
    }

    /// Generates one complete segment and verifies that every executed opcode
    /// reached its concrete replay producer.
    ///
    /// Keep the coverage check inside this concrete coordinator: the VM owns
    /// inventory order and lifetime fencing, while RV64IM owns its opcode-to-
    /// producer mapping. This avoids a generic trace-generator framework and
    /// makes skipping [`Self::finish`] impossible on the production entry path.
    pub fn generate_proving_ctx<VB>(
        self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        vm.generate_preflight_proving_ctx(
            self.program,
            self.transcript,
            self.replay_plan,
            self,
            |tracegen, chip| {
                tracegen
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
            |tracegen| {
                tracegen
                    .finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
        )
    }

    /// Generates one extension AIR in the VM inventory's normal reverse order.
    ///
    /// Replay producers update their shared lookup histograms. Periphery chips
    /// are then generated from those histograms through their ordinary
    /// record-independent path. Every other chip is known to be unexecuted by
    /// the constructor coverage check, so it receives a dummy trace.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        if let Some(chip) = chip.downcast_ref::<PhantomChipGPU>() {
            self.mark_generated([SystemOpcode::PHANTOM.global_opcode().as_usize() as u32]);
            return chip.generate_proving_ctx_from_postflight(
                self.program,
                self.transcript,
                self.replay_plan,
            );
        }

        macro_rules! replay_chip {
            ($chip_ty:ty, [$($opcode:expr),+ $(,)?]) => {
                if let Some(chip) = chip.downcast_ref::<$chip_ty>() {
                    self.mark_generated([$(
                        Self::opcode($opcode)
                    ),+]);
                    return chip.generate_proving_ctx_from_postflight(
                        self.program,
                        self.transcript,
                        self.replay_plan,
                    );
                }
            };
        }

        replay_chip!(AddSubChipGpu, [BaseAluOpcode::ADD, BaseAluOpcode::SUB]);
        replay_chip!(
            BitwiseLogicChipGpu,
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND,]
        );
        replay_chip!(AddSubWChipGpu, [BaseAluWOpcode::ADDW, BaseAluWOpcode::SUBW]);
        replay_chip!(LessThanChipGpu, [LessThanOpcode::SLT, LessThanOpcode::SLTU]);
        replay_chip!(ShiftLogicalChipGpu, [ShiftOpcode::SLL, ShiftOpcode::SRL]);
        replay_chip!(ShiftRightArithmeticChipGpu, [ShiftOpcode::SRA]);
        replay_chip!(
            ShiftWLogicalChipGpu,
            [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW]
        );
        replay_chip!(ShiftWRightArithmeticChipGpu, [ShiftWOpcode::SRAW]);
        replay_chip!(AddIWChipGpu, [BaseAluWImmOpcode::ADDIW]);
        replay_chip!(
            ShiftWLogicalImmChipGpu,
            [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW]
        );
        replay_chip!(ShiftWRightArithmeticImmChipGpu, [ShiftWImmOpcode::SRAIW]);
        replay_chip!(
            BranchEqualChipGpu,
            [BranchEqualOpcode::BEQ, BranchEqualOpcode::BNE]
        );
        replay_chip!(
            BranchLessThanChipGpu,
            [
                BranchLessThanOpcode::BLT,
                BranchLessThanOpcode::BLTU,
                BranchLessThanOpcode::BGE,
                BranchLessThanOpcode::BGEU,
            ]
        );
        replay_chip!(JalLuiChipGpu, [JalLuiOpcode::JAL, JalLuiOpcode::LUI]);
        replay_chip!(JalrChipGpu, [JalrOpcode::JALR]);
        replay_chip!(AuipcChipGpu, [AuipcOpcode::AUIPC]);
        replay_chip!(LoadSignExtendByteChipGpu, [LoadStoreOpcode::LOADB]);
        replay_chip!(LoadByteChipGpu, [LoadStoreOpcode::LOADBU]);
        replay_chip!(LoadSignExtendHalfwordChipGpu, [LoadStoreOpcode::LOADH]);
        replay_chip!(LoadHalfwordChipGpu, [LoadStoreOpcode::LOADHU]);
        replay_chip!(LoadSignExtendWordChipGpu, [LoadStoreOpcode::LOADW]);
        replay_chip!(LoadWordChipGpu, [LoadStoreOpcode::LOADWU]);
        replay_chip!(LoadDoublewordChipGpu, [LoadStoreOpcode::LOADD]);
        replay_chip!(StoreByteChipGpu, [LoadStoreOpcode::STOREB]);
        replay_chip!(StoreHalfwordChipGpu, [LoadStoreOpcode::STOREH]);
        replay_chip!(StoreWordChipGpu, [LoadStoreOpcode::STOREW]);
        replay_chip!(StoreDoublewordChipGpu, [LoadStoreOpcode::STORED]);
        replay_chip!(
            HintStoreChipGpu,
            [HintStoreOpcode::HINT_STORED, HintStoreOpcode::HINT_BUFFER,]
        );
        replay_chip!(MultiplicationChipGpu, [MulOpcode::MUL]);
        replay_chip!(MulWChipGpu, [MulWOpcode::MULW]);
        replay_chip!(
            MulHChipGpu,
            [MulHOpcode::MULH, MulHOpcode::MULHSU, MulHOpcode::MULHU]
        );
        replay_chip!(
            DivRemChipGpu,
            [
                DivRemOpcode::DIV,
                DivRemOpcode::DIVU,
                DivRemOpcode::REM,
                DivRemOpcode::REMU,
            ]
        );
        replay_chip!(
            DivRemWChipGpu,
            [
                DivRemWOpcode::DIVW,
                DivRemWOpcode::DIVUW,
                DivRemWOpcode::REMW,
                DivRemWOpcode::REMUW,
            ]
        );
        replay_chip!(AddIChipGpu, [BaseAluImmOpcode::ADDI]);
        replay_chip!(
            ShiftLogicalImmChipGpu,
            [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI]
        );
        replay_chip!(ShiftRightArithmeticImmChipGpu, [ShiftImmOpcode::SRAI]);
        replay_chip!(
            LessThanImmChipGpu,
            [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU]
        );
        replay_chip!(
            BitwiseLogicImmChipGpu,
            [
                BaseAluImmOpcode::XORI,
                BaseAluImmOpcode::ORI,
                BaseAluImmOpcode::ANDI,
            ]
        );
        if let Some(chip) = chip.downcast_ref::<Arc<VariableRangeCheckerChipGPU>>() {
            return Ok(
                <Arc<VariableRangeCheckerChipGPU> as Chip<GpuBackend>>::generate_proving_ctx(chip),
            );
        }
        if let Some(chip) = chip.downcast_ref::<Arc<BitwiseOperationLookupChipGPU<8>>>() {
            return Ok(<Arc<BitwiseOperationLookupChipGPU<8>> as Chip<
                GpuBackend,
            >>::generate_proving_ctx(chip));
        }
        if let Some(chip) = chip.downcast_ref::<Arc<RangeTupleCheckerChipGPU<2>>>() {
            return Ok(
                <Arc<RangeTupleCheckerChipGPU<2>> as Chip<GpuBackend>>::generate_proving_ctx(chip),
            );
        }
        if let Some(chip) = chip.downcast_ref::<Arc<Poseidon2PeripheryChipGPU>>() {
            return Ok(
                <Arc<Poseidon2PeripheryChipGPU> as Chip<GpuBackend>>::generate_proving_ctx(chip),
            );
        }

        Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()))
    }

    /// Completes one segment after every extension AIR context has been made.
    ///
    /// The safe VM trace-generation entry point performs the synchronized
    /// sticky-error read after every kernel has been submitted. This final
    /// check only proves that the reverse inventory walk visited a producer for
    /// every executed opcode.
    #[doc(hidden)]
    pub fn finish(self) -> Result<(), GpuPostflightError> {
        if !self.pending_opcodes.is_empty() {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "RV64IM preflight GPU tracegen did not visit chips for executed opcodes {:?}",
                self.pending_opcodes
            )));
        }
        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, Rv64I> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<AddSubAir>()?;
        let add_sub = AddSubChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<BitwiseLogicAir>()?;
        let bitwise_logic = BitwiseLogicChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(bitwise_logic);

        inventory.next_air::<AddSubWAir>()?;
        let add_sub_w = AddSubWChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(add_sub_w);

        inventory.next_air::<LessThanAir>()?;
        let lt = LessThanChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(lt);

        inventory.next_air::<ShiftLogicalAir>()?;
        let shift_logical = ShiftLogicalChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<ShiftRightArithmeticAir>()?;
        let shift_right_arithmetic =
            ShiftRightArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_right_arithmetic);

        inventory.next_air::<ShiftWLogicalAir>()?;
        let shift_w_logical = ShiftWLogicalChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_logical);

        inventory.next_air::<ShiftWRightArithmeticAir>()?;
        let shift_w_right_arithmetic =
            ShiftWRightArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_right_arithmetic);

        inventory.next_air::<AddIWAir>()?;
        let addi_w = AddIWChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(addi_w);

        inventory.next_air::<ShiftWLogicalImmAir>()?;
        let shift_w_logical_imm =
            ShiftWLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_logical_imm);

        inventory.next_air::<ShiftWRightArithmeticImmAir>()?;
        let shift_w_right_arithmetic_imm =
            ShiftWRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_right_arithmetic_imm);

        inventory.next_air::<LoadSignExtendByteAir>()?;
        let load_sign_extend_byte = LoadSignExtendByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_byte);

        inventory.next_air::<LoadByteAir>()?;
        let load_byte = LoadByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_byte);

        inventory.next_air::<StoreByteAir>()?;
        let store_byte = StoreByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_byte);

        inventory.next_air::<LoadSignExtendHalfwordAir>()?;
        let load_sign_extend_halfword = LoadSignExtendHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_halfword);

        inventory.next_air::<LoadHalfwordAir>()?;
        let load_halfword = LoadHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_halfword);

        inventory.next_air::<StoreHalfwordAir>()?;
        let store_halfword = StoreHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_halfword);

        inventory.next_air::<LoadSignExtendWordAir>()?;
        let load_sign_extend_word = LoadSignExtendWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend_word);

        inventory.next_air::<LoadWordAir>()?;
        let load_word = LoadWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_word);

        inventory.next_air::<StoreWordAir>()?;
        let store_word = StoreWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_word);

        inventory.next_air::<LoadDoublewordAir>()?;
        let load_doubleword = LoadDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_doubleword);

        inventory.next_air::<StoreDoublewordAir>()?;
        let store_doubleword = StoreDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(store_doubleword);

        inventory.next_air::<BranchEqualAir>()?;
        let beq = BranchEqualChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(beq);

        inventory.next_air::<BranchLessThanAir>()?;
        let blt = BranchLessThanChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(blt);

        inventory.next_air::<JalLuiAir>()?;
        let jal_lui = JalLuiChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<JalrAir>()?;
        let jalr = JalrChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jalr);

        inventory.next_air::<AuipcAir>()?;
        let auipc = AuipcChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(auipc);

        inventory.next_air::<AddIAir>()?;
        let addi = AddIChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(addi);

        inventory.next_air::<ShiftLogicalImmAir>()?;
        let shift_logical_imm =
            ShiftLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_logical_imm);

        inventory.next_air::<ShiftRightArithmeticImmAir>()?;
        let shift_right_arithmetic_imm =
            ShiftRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_right_arithmetic_imm);

        inventory.next_air::<LessThanImmAir>()?;
        let lt_imm = LessThanImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(lt_imm);

        inventory.next_air::<BitwiseLogicImmAir>()?;
        let bitwise_logic_imm = BitwiseLogicImmChipGpu::new(
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
impl VmProverExtension<GpuBabyBearPoseidon2Engine, Rv64M> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, GpuBackend>,
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
        inventory.next_air::<MultiplicationAir>()?;
        let mult = MultiplicationChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<MulWAir>()?;
        let mul_w = MulWChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_w);

        inventory.next_air::<MulHAir>()?;
        let mul_h = MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<DivRemAir>()?;
        let div_rem = DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(div_rem);

        inventory.next_air::<DivRemWAir>()?;
        let divrem_w = DivRemWChipGpu::new(
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
impl VmProverExtension<GpuBabyBearPoseidon2Engine, Rv64Io> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);

        inventory.next_air::<HintStoreAir>()?;
        let hint_store =
            HintStoreChipGpu::new(range_checker.clone(), byte_ptr_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}

#[cfg(test)]
mod history_tests;
#[cfg(all(test, feature = "rvr"))]
mod tests;
