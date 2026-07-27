use std::sync::Arc;

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, cuda_abi::UInt2,
    range_tuple::RangeTupleCheckerChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterCols, Rv64VecHeapAdapterRecord, Rv64VecHeapBranchU16AdapterCols,
    Rv64VecHeapBranchU16AdapterRecord, Rv64VecHeapU16AdapterCols, Rv64VecHeapU16AdapterRecord,
};
use openvm_riscv_circuit::{
    adapters::{RV64_BYTE_BITS, U16_BITS},
    AddSubCoreCols, AddSubCoreRecord, BitwiseLogicCoreCols, BitwiseLogicCoreRecord,
    BranchEqualCoreCols, BranchEqualCoreRecord, BranchLessThanCoreCols, BranchLessThanCoreRecord,
    LessThanCoreCols, LessThanCoreRecord, MultiplicationCoreCols, MultiplicationCoreRecord,
    ShiftLogicalCoreCols, ShiftLogicalCoreRecord, ShiftRightArithmeticCoreCols,
    ShiftRightArithmeticCoreRecord,
};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_bigint_transpiler::{
        Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
        Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
    },
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode,
    },
    openvm_riscv_transpiler::{
        BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode,
        ShiftOpcode,
    },
};

mod cuda_abi;

use crate::{INT256_NUM_MEMORY_BLOCKS, INT256_NUM_U16_LIMBS, INT256_NUM_U8_LIMBS, NUM_READS};

//////////////////////////////////////////////////////////////////////////////////////
/// AddSub (u16 limbs, range checker)
//////////////////////////////////////////////////////////////////////////////////////
pub type AddSub256AdapterRecord =
    Rv64VecHeapU16AdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
pub type AddSub256CoreRecord = AddSubCoreRecord<INT256_NUM_U16_LIMBS>;

#[derive(new)]
pub struct AddSub256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// BitwiseLogic (byte limbs, bitwise lookup)
//////////////////////////////////////////////////////////////////////////////////////
pub type BitwiseLogic256AdapterRecord =
    Rv64VecHeapAdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
pub type BitwiseLogic256CoreRecord = BitwiseLogicCoreRecord<INT256_NUM_U8_LIMBS>;

#[derive(new)]
pub struct BitwiseLogic256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// Branch Equal
//////////////////////////////////////////////////////////////////////////////////////
pub type BranchEqual256AdapterRecord =
    Rv64VecHeapBranchU16AdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS>;
pub type BranchEqual256CoreRecord = BranchEqualCoreRecord<INT256_NUM_U16_LIMBS>;

#[derive(new)]
pub struct BranchEqual256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// Less Than
//////////////////////////////////////////////////////////////////////////////////////
pub type LessThan256AdapterRecord = openvm_riscv_adapters::Rv64VecHeapU16AdapterRecord<
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS,
>;
pub type LessThan256CoreRecord = LessThanCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>;

#[derive(new)]
pub struct LessThan256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// Branch Less Than
//////////////////////////////////////////////////////////////////////////////////////
pub type BranchLessThan256AdapterRecord =
    Rv64VecHeapBranchU16AdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS>;
pub type BranchLessThan256CoreRecord = BranchLessThanCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>;

#[derive(new)]
pub struct BranchLessThan256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// Shift
//////////////////////////////////////////////////////////////////////////////////////
pub type ShiftLogical256U16AdapterRecord =
    Rv64VecHeapU16AdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
pub type ShiftRightArithmetic256AdapterRecord = ShiftLogical256U16AdapterRecord;
pub type ShiftLogical256CoreRecord = ShiftLogicalCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>;
pub type ShiftRightArithmetic256CoreRecord =
    ShiftRightArithmeticCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>;

#[derive(new)]
pub struct ShiftLogical256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

#[derive(new)]
pub struct ShiftRightArithmetic256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

//////////////////////////////////////////////////////////////////////////////////////
/// Multiplication
//////////////////////////////////////////////////////////////////////////////////////
pub type Multiplication256AdapterRecord =
    Rv64VecHeapAdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
pub type Multiplication256CoreRecord =
    MultiplicationCoreRecord<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>;

#[derive(new)]
pub struct Multiplication256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
fn int256_family_range<const N: usize>(
    replay_plan: &GpuPostflightPlan,
    opcodes: [openvm_instructions::VmOpcode; N],
) -> Result<std::ops::Range<usize>, GpuPostflightError> {
    let ranges = opcodes.map(|opcode| replay_plan.opcode_range(opcode));
    let Some(start) = ranges
        .iter()
        .filter(|range| !range.is_empty())
        .map(|r| r.start)
        .min()
    else {
        return Ok(0..0);
    };
    let end = ranges
        .iter()
        .filter(|range| !range.is_empty())
        .map(|range| range.end)
        .max()
        .unwrap();
    let count = ranges.iter().map(std::ops::Range::len).sum::<usize>();
    if end - start != count {
        return Err(GpuPostflightError::InvalidTranscript(
            "Int256 opcode ranges are not contiguous".to_string(),
        ));
    }
    Ok(start..end)
}

#[cfg(feature = "rvr")]
macro_rules! int256_replay_common_args {
    ($program:expr, $transcript:expr, $replay_plan:expr, $range:expr) => {
        (
            $program.instructions(),
            $program.pc_base(),
            $transcript.program_log(),
            $transcript.memory_log(),
            $transcript.initial_write_log(),
            $transcript.memory_predecessors(),
            $replay_plan.steps(),
            $range.start,
            $range.len(),
            $transcript.error_ptr(),
        )
    };
}

#[cfg(feature = "rvr")]
impl AddSub256ChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = int256_family_range(
            replay_plan,
            [
                Rv64BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
                Rv64BaseAlu256Opcode(BaseAluOpcode::SUB).global_opcode(),
            ],
        )?;
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = AddSubCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
            + Rv64VecHeapU16AdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let height = next_power_of_two_or_zero(range.len());
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let args = int256_replay_common_args!(program, transcript, replay_plan, range);
        unsafe {
            cuda_abi::replay::add_sub(
                trace.buffer(),
                height,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
                args.6,
                args.7,
                args.8,
                args.9,
                Rv64BaseAlu256Opcode::CLASS_OFFSET as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

#[cfg(feature = "rvr")]
impl BitwiseLogic256ChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = int256_family_range(
            replay_plan,
            [
                Rv64BaseAlu256Opcode(BaseAluOpcode::XOR).global_opcode(),
                Rv64BaseAlu256Opcode(BaseAluOpcode::OR).global_opcode(),
                Rv64BaseAlu256Opcode(BaseAluOpcode::AND).global_opcode(),
            ],
        )?;
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = BitwiseLogicCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64VecHeapAdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let height = next_power_of_two_or_zero(range.len());
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let args = int256_replay_common_args!(program, transcript, replay_plan, range);
        unsafe {
            cuda_abi::replay::bitwise(
                trace.buffer(),
                height,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
                args.6,
                args.7,
                args.8,
                args.9,
                Rv64BaseAlu256Opcode::CLASS_OFFSET as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

#[cfg(feature = "rvr")]
macro_rules! impl_int256_u16_replay {
    ($chip:ty, $width:expr, $opcodes:expr, $base:expr, $kind:expr) => {
        impl $chip {
            pub fn generate_proving_ctx_from_postflight(
                &self,
                program: &GpuPostflightProgram,
                transcript: &GpuPostflightTranscript,
                replay_plan: &GpuPostflightPlan,
            ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
                let device_ctx = &self.range_checker.device_ctx;
                program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
                let range = int256_family_range(replay_plan, $opcodes)?;
                if range.is_empty() {
                    return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
                }
                let width = $width;
                let height = next_power_of_two_or_zero(range.len());
                let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
                let args = int256_replay_common_args!(program, transcript, replay_plan, range);
                unsafe {
                    cuda_abi::replay::u16(
                        trace.buffer(),
                        height,
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                        args.6,
                        args.7,
                        args.8,
                        args.9,
                        $base as u32,
                        RV64_REGISTER_AS,
                        RV64_MEMORY_AS,
                        &self.range_checker.count,
                        self.pointer_max_bits as u32,
                        self.timestamp_max_bits as u32,
                        $kind,
                        device_ctx.stream.as_raw(),
                    )?;
                }
                Ok(AirProvingContext::simple_no_pis(trace))
            }
        }
    };
}

#[cfg(feature = "rvr")]
impl_int256_u16_replay!(
    LessThan256ChipGpu,
    LessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
        + Rv64VecHeapU16AdapterCols::<
            F,
            NUM_READS,
            INT256_NUM_MEMORY_BLOCKS,
            INT256_NUM_MEMORY_BLOCKS,
        >::width(),
    [
        Rv64LessThan256Opcode(LessThanOpcode::SLT).global_opcode(),
        Rv64LessThan256Opcode(LessThanOpcode::SLTU).global_opcode(),
    ],
    Rv64LessThan256Opcode::CLASS_OFFSET,
    cuda_abi::replay::U16Kind::LessThan
);

#[cfg(feature = "rvr")]
impl_int256_u16_replay!(
    ShiftLogical256ChipGpu,
    ShiftLogicalCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
        + Rv64VecHeapU16AdapterCols::<
            F,
            NUM_READS,
            INT256_NUM_MEMORY_BLOCKS,
            INT256_NUM_MEMORY_BLOCKS,
        >::width(),
    [
        Rv64Shift256Opcode(ShiftOpcode::SLL).global_opcode(),
        Rv64Shift256Opcode(ShiftOpcode::SRL).global_opcode(),
    ],
    Rv64Shift256Opcode::CLASS_OFFSET,
    cuda_abi::replay::U16Kind::ShiftLogical
);

#[cfg(feature = "rvr")]
impl_int256_u16_replay!(
    ShiftRightArithmetic256ChipGpu,
    ShiftRightArithmeticCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
        + Rv64VecHeapU16AdapterCols::<
            F,
            NUM_READS,
            INT256_NUM_MEMORY_BLOCKS,
            INT256_NUM_MEMORY_BLOCKS,
        >::width(),
    [Rv64Shift256Opcode(ShiftOpcode::SRA).global_opcode()],
    Rv64Shift256Opcode::CLASS_OFFSET,
    cuda_abi::replay::U16Kind::ShiftRightArithmetic
);

#[cfg(feature = "rvr")]
impl_int256_u16_replay!(
    BranchEqual256ChipGpu,
    BranchEqualCoreCols::<F, INT256_NUM_U16_LIMBS>::width()
        + Rv64VecHeapBranchU16AdapterCols::<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS>::width(),
    [
        Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
        Rv64BranchEqual256Opcode(BranchEqualOpcode::BNE).global_opcode(),
    ],
    Rv64BranchEqual256Opcode::CLASS_OFFSET,
    cuda_abi::replay::U16Kind::BranchEqual
);

#[cfg(feature = "rvr")]
impl_int256_u16_replay!(
    BranchLessThan256ChipGpu,
    BranchLessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
        + Rv64VecHeapBranchU16AdapterCols::<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS>::width(),
    [
        Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLT).global_opcode(),
        Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLTU).global_opcode(),
        Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BGE).global_opcode(),
        Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BGEU).global_opcode(),
    ],
    Rv64BranchLessThan256Opcode::CLASS_OFFSET,
    cuda_abi::replay::U16Kind::BranchLessThan
);

#[cfg(feature = "rvr")]
impl Multiplication256ChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = int256_family_range(
            replay_plan,
            [Rv64Mul256Opcode(MulOpcode::MUL).global_opcode()],
        )?;
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = MultiplicationCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64VecHeapAdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let height = next_power_of_two_or_zero(range.len());
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let args = int256_replay_common_args!(program, transcript, replay_plan, range);
        let sizes = self.range_tuple_checker.sizes;
        unsafe {
            cuda_abi::replay::multiplication(
                trace.buffer(),
                height,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
                args.6,
                args.7,
                args.8,
                args.9,
                Rv64Mul256Opcode::CLASS_OFFSET as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                &self.range_tuple_checker.count,
                UInt2 {
                    x: sizes[0],
                    y: sizes[1],
                },
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
