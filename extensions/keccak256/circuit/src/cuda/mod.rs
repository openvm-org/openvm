use std::{
    mem::size_of,
    sync::{Arc, Mutex},
};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, prover_backend::GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use p3_keccak_air::NUM_ROUNDS;

use crate::{
    keccakf_op::{columns::NUM_KECCAKF_OP_COLS, trace::KeccakfRecord, NUM_OP_ROWS_PER_INS},
    keccakf_perm::NUM_KECCAKF_PERM_COLS,
    xorin::{columns::NUM_XORIN_VM_COLS, trace::XorinVmRecordHeader},
};

mod cuda_abi;

// ========================== XorinVmChipGpu ==========================

#[derive(new)]
pub struct XorinVmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for XorinVmChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<XorinVmRecordHeader>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = NUM_XORIN_VM_COLS;
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::xorin::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

// ========================== Shared state for KeccakfOp <-> KeccakfPerm ==========================

/// Shared state to pass records from KeccakfOpChipGpu to KeccakfPermChipGpu
/// The OpChip generates first and stores the device buffer, then PermChip takes it.
#[derive(Default)]
pub struct SharedKeccakfRecords {
    /// Device buffer containing records (set by OpChip, consumed by PermChip)
    pub d_records: Option<DeviceBuffer<u8>>,
    /// Number of records
    pub num_records: usize,
}

pub type SharedKeccakfRecordsGpu = Arc<Mutex<SharedKeccakfRecords>>;

// ========================== KeccakfOpChipGpu ==========================

#[derive(new)]
pub struct KeccakfOpChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: u32,
    pub shared_records: SharedKeccakfRecordsGpu,
}

impl Chip<DenseRecordArena, GpuBackend> for KeccakfOpChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<KeccakfRecord>();
        let records = arena.allocated();
        if records.is_empty() {
            // Store empty state for PermChip
            let mut shared = self.shared_records.lock().unwrap();
            shared.d_records = None;
            shared.num_records = 0;
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let num_records = records.len() / RECORD_SIZE;
        let trace_width = NUM_KECCAKF_OP_COLS;
        let trace_height = next_power_of_two_or_zero(num_records * NUM_OP_ROWS_PER_INS);

        // Transfer records to GPU
        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::keccakf_op::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits,
            )
            .unwrap();
        }

        // Store records in shared state for PermChip
        {
            let mut shared = self.shared_records.lock().unwrap();
            shared.d_records = Some(d_records);
            shared.num_records = num_records;
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

// ========================== KeccakfPermChipGpu ==========================

#[derive(new)]
pub struct KeccakfPermChipGpu {
    pub shared_records: SharedKeccakfRecordsGpu,
}

impl Chip<DenseRecordArena, GpuBackend> for KeccakfPermChipGpu {
    fn generate_proving_ctx(&self, _arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        // Take records from shared state (set by OpChip)
        let (d_records, num_records) = {
            let mut shared = self.shared_records.lock().unwrap();
            (shared.d_records.take(), shared.num_records)
        };

        let Some(d_records) = d_records else {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        };

        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = NUM_KECCAKF_PERM_COLS;
        let trace_height = next_power_of_two_or_zero(num_records * NUM_ROUNDS);

        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::keccakf_perm::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
