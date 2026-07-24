use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, cuda_abi::UInt2,
    range_tuple::RangeTupleCheckerChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
#[cfg(not(feature = "rvr"))]
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
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

mod cuda_abi;

use crate::{INT256_NUM_MEMORY_BLOCKS, INT256_NUM_U16_LIMBS, INT256_NUM_U8_LIMBS, NUM_READS};

fn opaque_h2d<T>(
    records: &[T],
    segment_id: Option<u32>,
    device_ctx: &GpuDeviceCtx,
) -> DeviceBuffer<T> {
    #[cfg(feature = "rvr")]
    {
        openvm_circuit::arch::rvr::gpu_profile::opaque_h2d(records, segment_id, device_ctx)
    }
    #[cfg(not(feature = "rvr"))]
    {
        let _ = segment_id;
        records.to_device_on(device_ctx).unwrap()
    }
}

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

impl Chip<DenseRecordArena, GpuBackend> for AddSub256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(AddSub256AdapterRecord, AddSub256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = AddSubCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
            + Rv64VecHeapU16AdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::add_sub256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for BitwiseLogic256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BitwiseLogic256AdapterRecord, BitwiseLogic256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BitwiseLogicCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64VecHeapAdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::bitwise_logic256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for BranchEqual256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BranchEqual256AdapterRecord, BranchEqual256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchEqualCoreCols::<F, INT256_NUM_U16_LIMBS>::width()
            + Rv64VecHeapBranchU16AdapterCols::<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::beq256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for LessThan256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(LessThan256AdapterRecord, LessThan256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = LessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
            + openvm_riscv_adapters::Rv64VecHeapU16AdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::lt256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for BranchLessThan256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BranchLessThan256AdapterRecord, BranchLessThan256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchLessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
            + Rv64VecHeapBranchU16AdapterCols::<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::blt256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for ShiftLogical256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(ShiftLogical256U16AdapterRecord, ShiftLogical256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = ShiftLogicalCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
            + Rv64VecHeapU16AdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::shift256::tracegen_logical(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

impl Chip<DenseRecordArena, GpuBackend> for ShiftRightArithmetic256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            ShiftRightArithmetic256AdapterRecord,
            ShiftRightArithmetic256CoreRecord,
        )>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            ShiftRightArithmeticCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width()
                + Rv64VecHeapU16AdapterCols::<
                    F,
                    NUM_READS,
                    INT256_NUM_MEMORY_BLOCKS,
                    INT256_NUM_MEMORY_BLOCKS,
                >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            cuda_abi::shift256::tracegen_right_arithmetic(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
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

impl Chip<DenseRecordArena, GpuBackend> for Multiplication256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(Multiplication256AdapterRecord, Multiplication256CoreRecord)>();
        let g2_segment_id = arena.rvr_g2_segment_id();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = MultiplicationCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64VecHeapAdapterCols::<
                F,
                NUM_READS,
                INT256_NUM_MEMORY_BLOCKS,
                INT256_NUM_MEMORY_BLOCKS,
            >::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = opaque_h2d(records, g2_segment_id, device_ctx);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        let sizes = self.range_tuple_checker.sizes;
        let d_sizes = UInt2 {
            x: sizes[0],
            y: sizes[1],
        };
        unsafe {
            cuda_abi::mul256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                &self.range_tuple_checker.count,
                d_sizes,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
