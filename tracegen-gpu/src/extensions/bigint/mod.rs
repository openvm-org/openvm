use std::{mem::size_of, sync::Arc};

use openvm_circuit::{
    arch::{DenseRecordArena, VmAirWrapper},
    utils::next_power_of_two_or_zero,
};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterRecord,
    Rv32VecHeapAdapterRecord,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreAir, BaseAluCoreRecord, BranchEqualCoreAir, BranchEqualCoreRecord,
    BranchLessThanCoreAir, BranchLessThanCoreRecord, LessThanCoreAir, LessThanCoreRecord,
    MultiplicationCoreAir, MultiplicationCoreRecord, ShiftCoreAir, ShiftCoreRecord,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip, UInt2,
};

pub mod cuda;

#[cfg(test)]
mod tests;

pub type Rv32BaseAlu256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type BaseAlu256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type BaseAlu256CoreRecord = BaseAluCoreRecord<INT256_NUM_LIMBS>;

#[derive(new)]
pub struct BaseAlu256ChipGpu<'a> {
    pub air: Rv32BaseAlu256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for BaseAlu256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(BaseAlu256AdapterRecord, BaseAlu256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for BaseAlu256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::alu256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

pub type Rv32BranchEqual256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;

pub type BranchEqual256AdapterRecord = Rv32HeapBranchAdapterRecord<2>;
pub type BranchEqual256CoreRecord = BranchEqualCoreRecord<INT256_NUM_LIMBS>;

#[derive(new)]
pub struct BranchEqual256ChipGpu<'a> {
    pub air: Rv32BranchEqual256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for BranchEqual256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize =
            size_of::<(BranchEqual256AdapterRecord, BranchEqual256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for BranchEqual256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::beq256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

pub type Rv32LessThan256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type LessThan256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type LessThan256CoreRecord = LessThanCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct LessThan256ChipGpu<'a> {
    pub air: Rv32LessThan256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for LessThan256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(LessThan256AdapterRecord, LessThan256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for LessThan256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::lt256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

pub type Rv32BranchLessThan256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type BranchLessThan256AdapterRecord = Rv32HeapBranchAdapterRecord<2>;
pub type BranchLessThan256CoreRecord = BranchLessThanCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct BranchLessThan256ChipGpu<'a> {
    pub air: Rv32BranchLessThan256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for BranchLessThan256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize =
            size_of::<(BranchLessThan256AdapterRecord, BranchLessThan256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for BranchLessThan256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::blt256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

pub type Rv32Shift256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Shift256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type Shift256CoreRecord = ShiftCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct Shift256ChipGpu<'a> {
    pub air: Rv32Shift256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for Shift256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(Shift256AdapterRecord, Shift256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Shift256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::shift256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

pub type Rv32Multiplication256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Multiplication256AdapterRecord =
    Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;
pub type Multiplication256CoreRecord = MultiplicationCoreRecord<INT256_NUM_LIMBS, RV32_CELL_BITS>;

#[derive(new)]
pub struct Multiplication256ChipGpu<'a> {
    pub air: Rv32Multiplication256Air,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for Multiplication256ChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize =
            size_of::<(Multiplication256AdapterRecord, Multiplication256CoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Multiplication256ChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        let sizes = self.range_tuple_checker.air.bus.sizes;
        let d_sizes = UInt2 {
            x: sizes[0],
            y: sizes[1],
        };
        unsafe {
            cuda::mul256::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                &self.range_tuple_checker.count,
                d_sizes,
            )
            .unwrap();
        }
        trace
    }
}
