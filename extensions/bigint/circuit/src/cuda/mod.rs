use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix,
    chip::{get_empty_air_proving_ctx, UInt2},
    prelude::F,
    prover_backend::GpuBackend,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_rv32_adapters::{
    Rv32HeapBranchAdapterCols, Rv32HeapBranchAdapterRecord, Rv32VecHeapAdapterCols,
    Rv32VecHeapAdapterRecord,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreCols, BaseAluCoreRecord, BranchEqualCoreCols, BranchEqualCoreRecord,
    BranchLessThanCoreCols, BranchLessThanCoreRecord, LessThanCoreCols, LessThanCoreRecord,
    MultiplicationCoreCols, MultiplicationCoreRecord, ShiftCoreCols, ShiftCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

mod cuda_abi;

//////////////////////////////////////////////////////////////////////////////////////
/// ALU
//////////////////////////////////////////////////////////////////////////////////////
pub type BaseAlu256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type BaseAlu256CoreRecord = BaseAluCoreRecord<INT256_NUM_LIMBS>;

#[derive(new)]
pub struct BaseAlu256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for BaseAlu256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(BaseAlu256AdapterRecord, BaseAlu256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BaseAluCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32VecHeapAdapterCols::<F, 2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::alu256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Branch Equal
//////////////////////////////////////////////////////////////////////////////////////
pub type BranchEqual256AdapterRecord = Rv32HeapBranchAdapterRecord<2>;
pub type BranchEqual256CoreRecord = BranchEqualCoreRecord<INT256_NUM_LIMBS>;

#[derive(new)]
pub struct BranchEqual256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for BranchEqual256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BranchEqual256AdapterRecord, BranchEqual256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchEqualCoreCols::<F, INT256_NUM_LIMBS>::width()
            + Rv32HeapBranchAdapterCols::<F, 2, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::beq256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Less Than
//////////////////////////////////////////////////////////////////////////////////////
pub type LessThan256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type LessThan256CoreRecord = LessThanCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct LessThan256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for LessThan256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(LessThan256AdapterRecord, LessThan256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = LessThanCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32VecHeapAdapterCols::<F, 2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::lt256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Branch Less Than
//////////////////////////////////////////////////////////////////////////////////////
pub type BranchLessThan256AdapterRecord = Rv32HeapBranchAdapterRecord<2>;
pub type BranchLessThan256CoreRecord = BranchLessThanCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct BranchLessThan256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for BranchLessThan256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BranchLessThan256AdapterRecord, BranchLessThan256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchLessThanCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32HeapBranchAdapterCols::<F, 2, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::blt256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Shift
//////////////////////////////////////////////////////////////////////////////////////
pub type Shift256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type Shift256CoreRecord = ShiftCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct Shift256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Shift256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Shift256AdapterRecord, Shift256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = ShiftCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32VecHeapAdapterCols::<F, 2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::shift256::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
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
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type Multiplication256CoreRecord = MultiplicationCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct Multiplication256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Multiplication256ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(Multiplication256AdapterRecord, Multiplication256CoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = MultiplicationCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32VecHeapAdapterCols::<F, 2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

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
                RV32_CELL_BITS,
                &self.range_tuple_checker.count,
                d_sizes,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
