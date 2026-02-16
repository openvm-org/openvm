use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::{align_of, size_of},
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, CustomBorrow, ExecutionError, MultiRowLayout, MultiRowMetadata,
        PreflightExecutor, RecordArena, SizedRecord, TraceFiller, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::DeferralOutputCols;

#[derive(Clone, Copy, Debug, Default)]
pub struct DeferralOutputMetadata {
    pub num_rows: usize,
}

impl MultiRowMetadata for DeferralOutputMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_rows
    }
}

pub(crate) type DeferralOutputRecordLayout = MultiRowLayout<DeferralOutputMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone, Copy)]
pub struct DeferralOutputRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub num_rows: u32,
}

pub struct DeferralOutputRecordMut<'a, F> {
    pub record: &'a mut DeferralOutputRecord,
    _phantom: PhantomData<F>,
}

impl<'a, F> CustomBorrow<'a, DeferralOutputRecordMut<'a, F>, DeferralOutputRecordLayout> for [u8]
where
    F: PrimeField32,
{
    fn custom_borrow(
        &'a mut self,
        _layout: DeferralOutputRecordLayout,
    ) -> DeferralOutputRecordMut<'a, F> {
        let record: &mut DeferralOutputRecord =
            <[u8] as CustomBorrow<'a, &mut DeferralOutputRecord, ()>>::custom_borrow(self, ());
        DeferralOutputRecordMut {
            record,
            _phantom: PhantomData,
        }
    }

    unsafe fn extract_layout(&self) -> DeferralOutputRecordLayout {
        let record: &DeferralOutputRecord = self.borrow();
        DeferralOutputRecordLayout {
            metadata: DeferralOutputMetadata {
                num_rows: record.num_rows as usize,
            },
        }
    }
}

impl<'a, F> SizedRecord<DeferralOutputRecordLayout> for DeferralOutputRecordMut<'a, F>
where
    F: PrimeField32,
{
    fn size(layout: &DeferralOutputRecordLayout) -> usize {
        let row_width = DeferralOutputCols::<F>::width();
        layout.metadata.num_rows * row_width * size_of::<F>()
    }

    fn alignment(_layout: &DeferralOutputRecordLayout) -> usize {
        align_of::<DeferralOutputRecord>()
    }
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputExecutor<F> {
    /// Determines the number of trace rows to allocate for this opcode.
    /// The bound is enforced implicitly by the segment height.
    pub row_count_fn: fn(&Instruction<F>, &TracingMemory) -> usize,
}

#[derive(Clone, Debug, Default)]
pub struct DeferralOutputFiller;

impl<F, RA> PreflightExecutor<F, RA> for DeferralOutputExecutor<F>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, DeferralOutputRecordLayout, DeferralOutputRecordMut<'buf, F>>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::OUTPUT)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let num_rows = (self.row_count_fn)(instruction, state.memory);
        debug_assert!(num_rows > 0);

        let record = state
            .ctx
            .alloc(DeferralOutputRecordLayout::new(DeferralOutputMetadata {
                num_rows,
            }));
        record.record.from_pc = *state.pc;
        record.record.from_timestamp = state.memory.timestamp();
        record.record.num_rows = num_rows as u32;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F> TraceFiller<F> for DeferralOutputFiller
where
    F: PrimeField32,
{
    fn fill_trace(
        &self,
        _mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let row_width = trace_matrix.width;
        let mut trace = &mut trace_matrix.values[..];
        let mut rows_so_far = 0;

        let mut chunks = Vec::new();
        let mut sizes = Vec::new();

        loop {
            if rows_so_far >= rows_used {
                chunks.push(trace);
                sizes.push(0);
                break;
            }

            let (row0, _) = trace.split_at_mut(row_width);
            // SAFETY: row0 contains a valid record representation written by the executor.
            let mut row0 = row0;
            let record: &DeferralOutputRecord = unsafe { get_record_from_slice(&mut row0, ()) };
            let num_rows = record.num_rows as usize;
            debug_assert!(num_rows > 0);

            let (chunk, rest) = trace.split_at_mut(row_width * num_rows);
            chunks.push(chunk);
            sizes.push(num_rows);
            rows_so_far += num_rows;
            trace = rest;
        }

        for (chunk, num_rows) in chunks.into_iter().zip(sizes.into_iter()) {
            if num_rows == 0 {
                // padding rows
                for row in chunk.chunks_mut(row_width) {
                    unsafe {
                        std::ptr::write_bytes(
                            row.as_mut_ptr() as *mut u8,
                            0,
                            row_width * size_of::<F>(),
                        );
                    }
                }
                continue;
            }

            let mut record_from_pc = 0u32;
            let mut record_from_timestamp = 0u32;

            for (row_idx, mut row) in chunk.chunks_exact_mut(row_width).enumerate() {
                if row_idx == 0 {
                    // SAFETY: row contains a valid record representation written by the executor.
                    let record: &DeferralOutputRecord =
                        unsafe { get_record_from_slice(&mut row, ()) };
                    record_from_pc = record.from_pc;
                    record_from_timestamp = record.from_timestamp;
                }

                unsafe {
                    std::ptr::write_bytes(
                        row.as_mut_ptr() as *mut u8,
                        0,
                        row_width * size_of::<F>(),
                    );
                }

                let cols: &mut DeferralOutputCols<F> = row.borrow_mut();
                cols.is_active = F::ONE;
                cols.is_first_row = F::from_bool(row_idx == 0);
                cols.is_last_row = F::from_bool(row_idx + 1 == num_rows);

                if row_idx == 0 {
                    cols.from_state.pc = F::from_u32(record_from_pc);
                    cols.from_state.timestamp = F::from_u32(record_from_timestamp);
                }
            }
        }
    }
}
