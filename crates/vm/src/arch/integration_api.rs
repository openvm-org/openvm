use std::{
    any::type_name,
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    io::Cursor,
    marker::PhantomData,
    ptr::{copy_nonoverlapping, slice_from_raw_parts_mut},
    sync::Arc,
};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{
    execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
    ExecutionState, InsExecutorE1, InstructionExecutor, Result, Streams, VmStateMut,
};
use crate::system::memory::{
    online::{GuestMemory, TracingMemory},
    MemoryAuxColsFactory, MemoryController, SharedMemoryHelper,
};

/// The interface between primitive AIR and machine adapter AIR.
pub trait VmAdapterInterface<T> {
    /// The memory read data that should be exposed for downstream use
    type Reads;
    /// The memory write data that are expected to be provided by the integrator
    type Writes;
    /// The parts of the instruction that should be exposed to the integrator.
    /// This will typically include `is_valid`, which indicates whether the trace row
    /// is being used and `opcode` to indicate which opcode is being executed if the
    /// VmChip supports multiple opcodes.
    type ProcessedInstruction;
}

pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;

    /// [Air](openvm_stark_backend::p3_air::Air) constraints owned by the adapter.
    /// The `interface` is given as abstract expressions so it can be directly used in other AIR
    /// constraints.
    ///
    /// Adapters should document the max constraint degree as a function of the constraint degrees
    /// of `reads, writes, instruction`.
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        interface: AdapterAirContext<AB::Expr, Self::Interface>,
    );

    /// Return the `from_pc` expression.
    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var;
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    /// Returns `(to_pc, interface)`.
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I>;

    /// The offset the opcodes by this chip start from.
    /// This is usually just `CorrespondingOpcode::CLASS_OFFSET`,
    /// but sometimes (for modular chips, for example) it also depends on something else.
    fn start_offset(&self) -> usize;

    fn start_offset_expr(&self) -> AB::Expr {
        AB::Expr::from_canonical_usize(self.start_offset())
    }

    fn expr_to_global_expr(&self, local_expr: impl Into<AB::Expr>) -> AB::Expr {
        self.start_offset_expr() + local_expr.into()
    }

    fn opcode_to_global_expr(&self, local_opcode: impl LocalOpcode) -> AB::Expr {
        self.expr_to_global_expr(AB::Expr::from_canonical_usize(local_opcode.local_usize()))
    }
}

pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}

/// Given some minimum layout of type `Layout`, the `RecordArena` should allocate a buffer, of
/// size possibly larger than the record, and then return mutable pointers to the record within the
/// buffer.
pub trait RecordArena<'a, Layout, RecordMut> {
    /// Allocates underlying buffer and returns a mutable reference `RecordMut`.
    /// Note that calling this function may not call an underlying memory allocation as the record
    /// arena may be virtual.
    fn alloc(&'a mut self, layout: Layout) -> RecordMut;
}

/// Interface for trace generation of a single instruction.The trace is provided as a mutable
/// buffer during both instruction execution and trace generation.
/// It is expected that no additional memory allocation is necessary and the trace buffer
/// is sufficient, with possible overwriting.
pub trait TraceStep<F, CTX> {
    type RecordLayout;
    type RecordMut<'a>;

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>;

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    /// Displayable opcode name for logging and debugging purposes.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

// TODO[jpw]: this might be temporary trait before moving trace to CTX
pub trait RowMajorMatrixArena<F> {
    /// Set the arena's capacity based on the projected trace height.
    fn set_capacity(&mut self, trace_height: usize);
    fn with_capacity(height: usize, width: usize) -> Self;
    fn width(&self) -> usize;
    fn trace_offset(&self) -> usize;
    fn into_matrix(self) -> RowMajorMatrix<F>;
}

// TODO[jpw]: revisit if this trait makes sense
pub trait TraceFiller<F, CTX> {
    /// Populates `trace`. This function will always be called after
    /// [`TraceStep::execute`], so the `trace` should already contain the records necessary to fill
    /// in the rest of it.
    // TODO(ayush): come up with a better abstraction for chips that fill a dynamic number of rows
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync + Clone,
    {
        let width = trace.width();
        trace.values[..rows_used * width]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_trace_row(mem_helper, row_slice);
            });
        trace.values[rows_used * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_dummy_trace_row(mem_helper, row_slice);
            });
    }

    /// Populates `row_slice`. This function will always be called after
    /// [`TraceStep::execute`], so the `row_slice` should already contain context necessary to
    /// fill in the rest of the row. This function will be called for each row in the trace which
    /// is being used, and for all other rows in the trace see `fill_dummy_trace_row`.
    ///
    /// The provided `row_slice` will have length equal to the width of the AIR.
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        unreachable!("fill_trace_row is not implemented")
    }

    /// Populates `row_slice`. This function will be called on dummy rows.
    /// By default the trace is padded with empty (all 0) rows to make the height a power of 2.
    ///
    /// The provided `row_slice` will have length equal to the width of the AIR.
    fn fill_dummy_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // By default, the row is filled with zeroes
    }
}

/// Converts a field element slice into a record type.
/// This function transmutes the `&mut [F]` to raw bytes,
/// then uses the `CustomBorrow` trait to transmute to the desired record type `T`.
/// ## Safety
/// `slice` must satisfy the requirements of the `CustomBorrow` trait.
pub unsafe fn get_record_from_slice<'a, T, F, L>(slice: &mut &'a mut [F], layout: L) -> T
where
    [u8]: CustomBorrow<'a, T, L>,
{
    // The alignment of `[u8]` is always satisfiedƒ
    let record_buffer =
        &mut *slice_from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, size_of_val::<[F]>(*slice));
    let record: T = record_buffer.custom_borrow(layout);
    record
}

/// Minimal layout information that [RecordArena] requires for record allocation
/// in scenarios involving chips that:
/// - have a single row per record, and
/// - have trace row = [adapter_row, core_row]
/// **NOTE**: `M` is the metadata type that implements `AdapterCoreMetadata`
#[derive(Debug, Clone, Default)]
pub struct AdapterCoreLayout<M> {
    pub metadata: M,
}

/// `Metadata` types need to implement this trait to be used with `AdapterCoreLayout`
/// **NOTE**: get_adapter_width returns the size in bytes
pub trait AdapterCoreMetadata {
    fn get_adapter_width() -> usize;
}

impl<M> AdapterCoreLayout<M> {
    pub fn new() -> Self
    where
        M: Default,
    {
        Self::default()
    }

    pub fn with_metadata(metadata: M) -> Self {
        Self { metadata }
    }
}

/// Empty metadata that implements `AdapterCoreMetadata`
/// **NOTE**: `AS` is the adapter type that implements `AdapterTraceStep`
/// **WARNING**: `AS::WIDTH` is the number of field elements, not the size in bytes
pub struct AdapterCoreEmptyMetadata<F, AS> {
    _phantom: PhantomData<(F, AS)>,
}

impl<F, AS> Clone for AdapterCoreEmptyMetadata<F, AS> {
    fn clone(&self) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F, AS> AdapterCoreEmptyMetadata<F, AS> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F, AS> Default for AdapterCoreEmptyMetadata<F, AS> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F, AS> AdapterCoreMetadata for AdapterCoreEmptyMetadata<F, AS>
where
    AS: AdapterTraceStep<F, ()>,
{
    #[inline(always)]
    fn get_adapter_width() -> usize {
        AS::WIDTH * size_of::<F>()
    }
}

/// AdapterCoreLayout with empty metadata that can be used by chips that have record type
/// (&mut A, &mut C) where `A` and `C` are `Sized`
pub type EmptyAdapterCoreLayout<F, AS> = AdapterCoreLayout<AdapterCoreEmptyMetadata<F, AS>>;

/// Minimal layout information that [RecordArena] requires for record allocation
/// in scenarios involving chips that:
/// - can have multiple rows per record, and
/// - have possibly variable length records
/// **NOTE**: `M` is the metadata type that implements `MultiRowMetadata`
#[derive(Debug, Clone, Default, derive_new::new)]
pub struct MultiRowLayout<M> {
    pub metadata: M,
}

/// `Metadata` types need to implement this trait to be used with `MultiRowLayout`
pub trait MultiRowMetadata {
    fn get_num_rows(&self) -> usize;
}

/// Empty metadata that implements `MultiRowMetadata` with `get_num_rows` always returning 1
#[derive(Debug, Clone, Default, derive_new::new)]
pub struct EmptyMultiRowMetadata {}

impl MultiRowMetadata for EmptyMultiRowMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        1
    }
}

/// Empty metadata that implements `MultiRowMetadata`
pub type EmptyMultiRowLayout = MultiRowLayout<EmptyMultiRowMetadata>;

/// A trait that allows for custom implementation of `borrow` given the necessary information
/// This is useful for record structs that have dynamic size
pub trait CustomBorrow<'a, T, L> {
    fn custom_borrow(&'a mut self, layout: L) -> T;

    /// Given `&self` as a valid starting pointer of a reference that has already been previously
    /// allocated and written to, extracts and returns the corresponding layout.
    /// This must work even if `T` is not sized.
    ///
    /// # Safety
    /// - `&self` must be a valid starting pointer on which `custom_borrow` has already been called
    /// - The data underlying `&self` has already been written to and is self-describing, so layout
    ///   can be extracted
    unsafe fn extract_layout(&self) -> L;
}

/// If a struct implements `BorrowMut<T>`, then the same implementation can be used for
/// `CustomBorrow::custom_borrow` with any layout
impl<'a, T: Sized, L: Default> CustomBorrow<'a, &'a mut T, L> for [u8]
where
    [u8]: BorrowMut<T>,
{
    fn custom_borrow(&'a mut self, _layout: L) -> &'a mut T {
        self.borrow_mut()
    }

    unsafe fn extract_layout(&self) -> L {
        L::default()
    }
}

/// `SizedRecord` is a trait that provides additional information about the size and alignment
/// requirements of a record. Should be implemented on RecordMut types
pub trait SizedRecord<Layout> {
    /// The minimal size in bytes that the RecordMut requires to be properly constructed
    /// given the layout.
    fn size(layout: &Layout) -> usize;
    /// The minimal alignment required for the RecordMut to be properly constructed
    /// given the layout.
    fn alignment(layout: &Layout) -> usize;
}

impl<Layout, Record> SizedRecord<Layout> for &mut Record
where
    Record: Sized,
{
    fn size(_layout: &Layout) -> usize {
        size_of::<Record>()
    }

    fn alignment(_layout: &Layout) -> usize {
        align_of::<Record>()
    }
}

// TEMP[jpw]: buffer should be inside CTX
pub struct MatrixRecordArena<F> {
    pub trace_buffer: Vec<F>,
    // TODO(ayush): width should be a constant?
    pub width: usize,
    pub trace_offset: usize,
}

impl<F: Field> MatrixRecordArena<F> {
    pub fn alloc_single_row(&mut self) -> &mut [u8] {
        self.alloc_buffer(1)
    }

    pub fn alloc_buffer(&mut self, num_rows: usize) -> &mut [u8] {
        let start = self.trace_offset;
        self.trace_offset += num_rows * self.width;
        let row_slice = &mut self.trace_buffer[start..self.trace_offset];
        let size = size_of_val(row_slice);
        let ptr = row_slice as *mut [F] as *mut u8;
        // SAFETY:
        // - `ptr` is non-null
        // - `size` is correct
        // - alignment of `u8` is always satisfied
        unsafe { &mut *std::ptr::slice_from_raw_parts_mut(ptr, size) }
    }
}

impl<F: Field> RowMajorMatrixArena<F> for MatrixRecordArena<F> {
    fn set_capacity(&mut self, trace_height: usize) {
        let size = trace_height * self.width;
        // PERF: use memset
        self.trace_buffer.resize(size, F::ZERO);
    }

    fn with_capacity(height: usize, width: usize) -> Self {
        let trace_buffer = F::zero_vec(height * width);
        Self {
            trace_buffer,
            width,
            trace_offset: 0,
        }
    }

    fn width(&self) -> usize {
        self.width
    }

    fn trace_offset(&self) -> usize {
        self.trace_offset
    }

    fn into_matrix(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self.trace_buffer, self.width)
    }
}

/// [RecordArena] implementation for [MatrixRecordArena], with [AdapterCoreLayout]
/// **NOTE**: `A` is the adapter RecordMut type and `C` is the core RecordMut type
impl<'a, F: Field, A, C, M: AdapterCoreMetadata> RecordArena<'a, AdapterCoreLayout<M>, (A, C)>
    for MatrixRecordArena<F>
where
    [u8]: CustomBorrow<'a, A, AdapterCoreLayout<M>> + CustomBorrow<'a, C, AdapterCoreLayout<M>>,
    M: Clone,
{
    fn alloc(&'a mut self, layout: AdapterCoreLayout<M>) -> (A, C) {
        let adapter_width = M::get_adapter_width();
        let buffer = self.alloc_single_row();
        // Doing a unchecked split here for perf
        let (adapter_buffer, core_buffer) = unsafe { buffer.split_at_mut_unchecked(adapter_width) };

        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);

        (adapter_record, core_record)
    }
}

/// [RecordArena] implementation for [MatrixRecordArena], with [MultiRowLayout]
/// **NOTE**: `R` is the RecordMut type
impl<'a, F: Field, M: MultiRowMetadata, R> RecordArena<'a, MultiRowLayout<M>, R>
    for MatrixRecordArena<F>
where
    [u8]: CustomBorrow<'a, R, MultiRowLayout<M>>,
{
    fn alloc(&'a mut self, layout: MultiRowLayout<M>) -> R {
        let buffer = self.alloc_buffer(layout.metadata.get_num_rows());
        let record: R = buffer.custom_borrow(layout);
        record
    }
}

pub struct DenseRecordArena {
    pub records_buffer: Cursor<Vec<u8>>,
}

const MAX_ALIGNMENT: usize = 32;

impl DenseRecordArena {
    /// Creates a new [DenseRecordArena] with the given capacity in bytes.
    pub fn with_capacity(size_bytes: usize) -> Self {
        let buffer = vec![0; size_bytes + MAX_ALIGNMENT];
        let offset = (MAX_ALIGNMENT - (buffer.as_ptr() as usize % MAX_ALIGNMENT)) % MAX_ALIGNMENT;
        let mut cursor = Cursor::new(buffer);
        cursor.set_position(offset as u64);
        Self {
            records_buffer: cursor,
        }
    }

    pub fn set_capacity(&mut self, size_bytes: usize) {
        let buffer = vec![0; size_bytes + MAX_ALIGNMENT];
        let offset = (MAX_ALIGNMENT - (buffer.as_ptr() as usize % MAX_ALIGNMENT)) % MAX_ALIGNMENT;
        let mut cursor = Cursor::new(buffer);
        cursor.set_position(offset as u64);
        self.records_buffer = cursor;
    }

    /// Allocates a single record of the given type and returns a mutable reference to it.
    pub fn alloc_one<'a, T>(&mut self) -> &'a mut T {
        let begin = self.records_buffer.position();
        let width = size_of::<T>();
        debug_assert!(begin as usize + width <= self.records_buffer.get_ref().len());
        self.records_buffer.set_position(begin + width as u64);
        unsafe {
            &mut *(self
                .records_buffer
                .get_mut()
                .as_mut_ptr()
                .add(begin as usize) as *mut T)
        }
    }

    /// Allocates a slice of records of the given type and returns a mutable reference to it.
    pub fn alloc_many<'a, T>(&mut self, count: usize) -> &'a mut [T] {
        let begin = self.records_buffer.position();
        let width = size_of::<T>() * count;
        debug_assert!(begin as usize + width <= self.records_buffer.get_ref().len());
        self.records_buffer.set_position(begin + width as u64);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.records_buffer
                    .get_mut()
                    .as_mut_ptr()
                    .add(begin as usize) as *mut T,
                count,
            )
        }
    }

    pub fn alloc_bytes<'a>(&mut self, count: usize) -> &'a mut [u8] {
        self.alloc_many::<u8>(count)
    }

    pub fn allocated(&self) -> &[u8] {
        let size = self.records_buffer.position() as usize;
        let offset = (MAX_ALIGNMENT
            - (self.records_buffer.get_ref().as_ptr() as usize % MAX_ALIGNMENT))
            % MAX_ALIGNMENT;
        &self.records_buffer.get_ref()[offset..size]
    }

    pub fn allocated_mut(&mut self) -> &mut [u8] {
        let size = self.records_buffer.position() as usize;
        let offset = (MAX_ALIGNMENT
            - (self.records_buffer.get_ref().as_ptr() as usize % MAX_ALIGNMENT))
            % MAX_ALIGNMENT;
        &mut self.records_buffer.get_mut()[offset..size]
    }

    pub fn align_to(&mut self, alignment: usize) {
        debug_assert!(MAX_ALIGNMENT % alignment == 0);
        let offset =
            (alignment - (self.records_buffer.get_ref().as_ptr() as usize % alignment)) % alignment;
        self.records_buffer.set_position(offset as u64);
    }

    // Returns a [RecordSeeker] on the allocated buffer
    pub fn get_record_seeker<'a, R, L>(&'a mut self) -> RecordSeeker<'a, DenseRecordArena, R, L> {
        RecordSeeker::new(self.allocated_mut())
    }
}

/// [RecordArena] implementation for [DenseRecordArena], with [AdapterCoreLayout]
/// **NOTE**: `A` is the adapter RecordMut type and `C` is the core record type
impl<'a, A, C, M> RecordArena<'a, AdapterCoreLayout<M>, (A, C)> for DenseRecordArena
where
    [u8]: CustomBorrow<'a, A, AdapterCoreLayout<M>> + CustomBorrow<'a, C, AdapterCoreLayout<M>>,
    M: Clone,
    A: SizedRecord<AdapterCoreLayout<M>>,
    C: SizedRecord<AdapterCoreLayout<M>>,
{
    fn alloc(&'a mut self, layout: AdapterCoreLayout<M>) -> (A, C) {
        let adapter_alignment = A::alignment(&layout);
        let core_alignment = C::alignment(&layout);
        let adapter_size = A::size(&layout);
        let aligned_adapter_size = adapter_size.next_multiple_of(core_alignment);
        let core_size = C::size(&layout);
        let aligned_core_size = (aligned_adapter_size + core_size)
            .next_multiple_of(adapter_alignment)
            - aligned_adapter_size;
        debug_assert_eq!(MAX_ALIGNMENT % adapter_alignment, 0);
        debug_assert_eq!(MAX_ALIGNMENT % core_alignment, 0);
        let buffer = self.alloc_bytes(aligned_adapter_size + aligned_core_size);
        // Doing an unchecked split here for perf
        let (adapter_buffer, core_buffer) =
            unsafe { buffer.split_at_mut_unchecked(aligned_adapter_size) };

        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);

        (adapter_record, core_record)
    }
}

/// [RecordArena] implementation for [DenseRecordArena], with [MultiRowLayout]
/// **NOTE**: `R` is the RecordMut type
impl<'a, R, M> RecordArena<'a, MultiRowLayout<M>, R> for DenseRecordArena
where
    [u8]: CustomBorrow<'a, R, MultiRowLayout<M>>,
    R: SizedRecord<MultiRowLayout<M>>,
{
    fn alloc(&'a mut self, layout: MultiRowLayout<M>) -> R {
        let record_size = R::size(&layout);
        let record_alignment = R::alignment(&layout);
        let aligned_record_size = record_size.next_multiple_of(record_alignment);
        let buffer = self.alloc_bytes(aligned_record_size);
        let record: R = buffer.custom_borrow(layout);
        record
    }
}

// This is a helper struct that implements a few utility methods
pub struct RecordSeeker<'a, RA, RecordMut, Layout> {
    pub buffer: &'a mut [u8], // The buffer that the records are written to
    _phantom: PhantomData<(RA, RecordMut, Layout)>,
}

impl<'a, RA, RecordMut, Layout> RecordSeeker<'a, RA, RecordMut, Layout> {
    pub fn new(record_buffer: &'a mut [u8]) -> Self {
        Self {
            buffer: record_buffer,
            _phantom: PhantomData,
        }
    }
}

// `RecordSeeker` implementation for [DenseRecordArena], with [MultiRowLayout]
// **NOTE** Assumes that `layout` can be extracted from the record alone
impl<'a, R, M> RecordSeeker<'a, DenseRecordArena, R, MultiRowLayout<M>>
where
    [u8]: CustomBorrow<'a, R, MultiRowLayout<M>>,
    R: SizedRecord<MultiRowLayout<M>>,
    M: MultiRowMetadata + Clone,
{
    // Returns the layout at the given offset in the buffer
    // **SAFETY**: `offset` has to be a valid offset, pointing to the start of a record
    pub fn get_layout_at(offset: &mut usize, buffer: &[u8]) -> MultiRowLayout<M> {
        let buffer = &buffer[*offset..];
        unsafe { buffer.extract_layout() }
    }

    // Returns a record at the given offset in the buffer
    // **SAFETY**: `offset` has to be a valid offset, pointing to the start of a record
    pub fn get_record_at(offset: &mut usize, buffer: &'a mut [u8]) -> R {
        let layout = Self::get_layout_at(offset, buffer);
        let buffer = &mut buffer[*offset..];
        let record_size = R::size(&layout);
        let record_alignment = R::alignment(&layout);
        let aligned_record_size = record_size.next_multiple_of(record_alignment);
        let record: R = buffer.custom_borrow(layout);
        *offset += aligned_record_size;
        record
    }

    // Returns a vector of all the records in the buffer
    pub fn extract_records(&'a mut self) -> Vec<R> {
        let mut records = Vec::new();
        let len = self.buffer.len();
        let buff = &mut self.buffer[..];
        let mut offset = 0;
        while offset < len {
            let record: R = {
                let buff = unsafe { &mut *slice_from_raw_parts_mut(buff.as_mut_ptr(), len) };
                Self::get_record_at(&mut offset, buff)
            };
            records.push(record);
        }
        records
    }

    // Transfers the records in the buffer to a [MatrixRecordArena], used in testing
    pub fn transfer_to_matrix_arena<F: PrimeField32>(
        &'a mut self,
        arena: &mut MatrixRecordArena<F>,
    ) {
        let len = self.buffer.len();
        arena.trace_offset = 0;
        let mut offset = 0;
        while offset < len {
            let layout = Self::get_layout_at(&mut offset, self.buffer);
            let record_size = R::size(&layout);
            let record_alignment = R::alignment(&layout);
            let aligned_record_size = record_size.next_multiple_of(record_alignment);
            let src_ptr = unsafe { self.buffer.as_ptr().add(offset) };
            let dst_ptr = arena
                .alloc_buffer(layout.metadata.get_num_rows())
                .as_mut_ptr();
            unsafe { copy_nonoverlapping(src_ptr, dst_ptr, aligned_record_size) };
            offset += aligned_record_size;
        }
    }
}

// `RecordSeeker` implementation for [DenseRecordArena], with [AdapterCoreLayout]
// **NOTE** Assumes that `layout` is the same for all the records, so it is expected to be passed as
// a parameter
impl<'a, A, C, M> RecordSeeker<'a, DenseRecordArena, (A, C), AdapterCoreLayout<M>>
where
    [u8]: CustomBorrow<'a, A, AdapterCoreLayout<M>> + CustomBorrow<'a, C, AdapterCoreLayout<M>>,
    A: SizedRecord<AdapterCoreLayout<M>>,
    C: SizedRecord<AdapterCoreLayout<M>>,
    M: AdapterCoreMetadata + Clone,
{
    // A utility function to get the aligned widths of the adapter and core records
    fn get_aligned_sizes(layout: &AdapterCoreLayout<M>) -> (usize, usize) {
        let adapter_alignment = A::alignment(&layout);
        let core_alignment = C::alignment(&layout);
        let adapter_size = A::size(&layout);
        let aligned_adapter_size = adapter_size.next_multiple_of(core_alignment);
        let core_size = C::size(&layout);
        let aligned_core_size = (aligned_adapter_size + core_size)
            .next_multiple_of(adapter_alignment)
            - aligned_adapter_size;
        (aligned_adapter_size, aligned_core_size)
    }

    // Returns a record at the given offset in the buffer
    // **SAFETY**: `offset` has to be a valid offset, pointing to the start of a record
    pub fn get_record_at(
        offset: &mut usize,
        buffer: &'a mut [u8],
        layout: AdapterCoreLayout<M>,
    ) -> (A, C) {
        let buffer = &mut buffer[*offset..];
        let (adapter_size, core_size) = Self::get_aligned_sizes(&layout);
        let (adapter_buffer, core_buffer) = unsafe { buffer.split_at_mut_unchecked(adapter_size) };
        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);
        *offset += adapter_size + core_size;
        (adapter_record, core_record)
    }

    // Returns a vector of all the records in the buffer
    pub fn extract_records(&'a mut self, layout: AdapterCoreLayout<M>) -> Vec<(A, C)> {
        let mut records = Vec::new();
        let len = self.buffer.len();
        let buff = &mut self.buffer[..];
        let mut offset = 0;
        while offset < len {
            let record: (A, C) = {
                let buff = unsafe { &mut *slice_from_raw_parts_mut(buff.as_mut_ptr(), len) };
                Self::get_record_at(&mut offset, buff, layout.clone())
            };
            records.push(record);
        }
        records
    }

    // Transfers the records in the buffer to a [MatrixRecordArena], used in testing
    pub fn transfer_to_matrix_arena<F: PrimeField32>(
        &'a mut self,
        arena: &mut MatrixRecordArena<F>,
        layout: AdapterCoreLayout<M>,
    ) {
        let len = self.buffer.len();
        arena.trace_offset = 0;
        let mut offset = 0;
        let (adapter_size, core_size) = Self::get_aligned_sizes(&layout);
        while offset < len {
            let dst_buffer = arena.alloc_single_row();
            let (adapter_buf, core_buf) =
                unsafe { dst_buffer.split_at_mut_unchecked(M::get_adapter_width()) };
            unsafe {
                let src_ptr = self.buffer.as_ptr().add(offset);
                copy_nonoverlapping(src_ptr, adapter_buf.as_mut_ptr(), adapter_size);
                copy_nonoverlapping(src_ptr.add(adapter_size), core_buf.as_mut_ptr(), core_size);
            }
            offset += adapter_size + core_size;
        }
    }
}

// TODO(ayush): rename to ChipWithExecutionContext or something
pub struct NewVmChipWrapper<F, AIR, STEP, RA> {
    pub air: AIR,
    pub step: STEP,
    pub arena: RA,
    mem_helper: SharedMemoryHelper<F>,
}

// TODO(AG): more general RA
impl<F, AIR, STEP> NewVmChipWrapper<F, AIR, STEP, MatrixRecordArena<F>>
where
    F: Field,
    AIR: BaseAir<F>,
{
    pub fn new(air: AIR, step: STEP, mem_helper: SharedMemoryHelper<F>) -> Self {
        let width = air.width();
        assert!(
            align_of::<F>() >= align_of::<u32>(),
            "type {} should have at least alignment of u32",
            type_name::<F>()
        );
        let arena = MatrixRecordArena::with_capacity(0, width);
        Self {
            air,
            step,
            arena,
            mem_helper,
        }
    }

    pub fn set_trace_buffer_height(&mut self, height: usize) {
        self.arena.set_capacity(height);
    }
}

// TODO(AG): more general RA
impl<F, AIR, STEP> NewVmChipWrapper<F, AIR, STEP, DenseRecordArena>
where
    F: Field,
    AIR: BaseAir<F>,
{
    pub fn new(air: AIR, step: STEP, mem_helper: SharedMemoryHelper<F>) -> Self {
        assert!(
            align_of::<F>() >= align_of::<u32>(),
            "type {} should have at least alignment of u32",
            type_name::<F>()
        );
        let arena = DenseRecordArena::with_capacity(0);
        Self {
            air,
            step,
            arena,
            mem_helper,
        }
    }

    pub fn set_trace_buffer_height(&mut self, height: usize) {
        let width = self.air.width();
        self.arena.set_capacity(height * width * size_of::<F>());
    }
}

impl<F, AIR, STEP, RA> InstructionExecutor<F> for NewVmChipWrapper<F, AIR, STEP, RA>
where
    F: PrimeField32,
    STEP: TraceStep<F, ()> // TODO: CTX?
        + StepExecutorE1<F>,
    for<'buf> RA: RecordArena<'buf, STEP::RecordLayout, STEP::RecordMut<'buf>>,
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        let mut pc = from_state.pc;
        let state = VmStateMut {
            pc: &mut pc,
            memory: &mut memory.memory,
            streams,
            rng,
            ctx: &mut (),
        };
        self.step.execute(state, instruction, &mut self.arena)?;

        Ok(ExecutionState {
            pc,
            timestamp: memory.memory.timestamp,
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.step.get_opcode_name(opcode)
    }
}

// Note[jpw]: the statement we want is:
// - `Air` is an `Air<AB>` for all `AB: AirBuilder`s needed by stark-backend
// which is equivalent to saying it implements AirRef<SC>
// The where clauses to achieve this statement is unfortunately really verbose.
impl<SC, AIR, STEP, RA> Chip<SC> for NewVmChipWrapper<Val<SC>, AIR, STEP, RA>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    STEP: TraceStep<Val<SC>, ()> + TraceFiller<Val<SC>, ()> + Send + Sync,
    AIR: Clone + AnyRap<SC> + 'static,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let width = self.arena.width();
        assert_eq!(self.arena.trace_offset() % width, 0);
        let rows_used = self.arena.trace_offset() / width;
        let height = next_power_of_two_or_zero(rows_used);
        let mut trace = self.arena.into_matrix();
        // This should be automatic since trace_buffer's height is a power of two:
        assert!(height.checked_mul(width).unwrap() <= trace.values.len());
        trace.values.truncate(height * width);
        let mem_helper = self.mem_helper.as_borrowed();
        self.step.fill_trace(&mem_helper, &mut trace, rows_used);
        drop(self.mem_helper);

        AirProofInput::simple(trace, self.step.generate_public_values())
    }
}

impl<F, AIR, C, RA> ChipUsageGetter for NewVmChipWrapper<F, AIR, C, RA>
where
    C: Sync,
    RA: RowMajorMatrixArena<F>,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.arena.trace_offset() / self.arena.width()
    }
    fn trace_width(&self) -> usize {
        self.arena.width()
    }
}

/// A helper trait for expressing generic state accesses within the implementation of
/// [TraceStep]. Note that this is only a helper trait when the same interface of state access
/// is reused or shared by multiple implementations. It is not required to implement this trait if
/// it is easier to implement the [TraceStep] trait directly without this trait.
pub trait AdapterTraceStep<F, CTX> {
    const WIDTH: usize;
    type ReadData;
    type WriteData;
    // @dev This can either be a &mut _ type or a struct with &mut _ fields.
    // The latter is helpful if we want to directly write certain values in place into a trace
    // matrix.
    type RecordMut<'a>
    where
        Self: 'a;

    fn start(pc: u32, memory: &TracingMemory<F>, record: &mut Self::RecordMut<'_>);

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData;

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    );
}

// NOTE[jpw]: cannot reuse `TraceSubRowGenerator` trait because we need associated constant
// `WIDTH`.
pub trait AdapterTraceFiller<F, CTX>: AdapterTraceStep<F, CTX> {
    /// Post-execution filling of rest of adapter row.
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]);
}

pub trait AdapterExecutorE1<F>
where
    F: PrimeField32,
{
    type ReadData;
    type WriteData;

    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx;

    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx;
}

// TODO: Rename core/step to operator
pub trait StepExecutorE1<F> {
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx;

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>;
}

impl<F, A, S> InsExecutorE1<F> for NewVmChipWrapper<F, A, S, MatrixRecordArena<F>>
where
    F: PrimeField32,
    S: StepExecutorE1<F>,
    A: BaseAir<F>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        self.step.execute_e1(state, instruction)
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>
    where
        F: PrimeField32,
    {
        self.step.execute_metered(state, instruction, chip_index)
    }

    fn set_trace_height(&mut self, height: usize) {
        self.set_trace_buffer_height(height);
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}

impl<F, A, C> BaseAir<F> for VmAirWrapper<A, C>
where
    A: BaseAir<F>,
    C: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.adapter.width() + self.core.width()
    }
}

impl<F, A, M> BaseAirWithPublicValues<F> for VmAirWrapper<A, M>
where
    A: BaseAir<F>,
    M: BaseAirWithPublicValues<F>,
{
    fn num_public_values(&self) -> usize {
        self.core.num_public_values()
    }
}

// Current cached trace is not supported
impl<F, A, M> PartitionedBaseAir<F> for VmAirWrapper<A, M>
where
    A: BaseAir<F>,
    M: BaseAir<F>,
{
}

impl<AB, A, M> Air<AB> for VmAirWrapper<A, M>
where
    AB: AirBuilder,
    A: VmAdapterAir<AB>,
    M: VmCoreAir<AB, A::Interface>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let (local_adapter, local_core) = local.split_at(self.adapter.width());

        let ctx = self
            .core
            .eval(builder, local_core, self.adapter.get_from_pc(local_adapter));
        self.adapter.eval(builder, local_adapter, ctx);
    }
}

// =================================================================================================
// Concrete adapter interfaces
// =================================================================================================

/// The most common adapter interface.
/// Performs `NUM_READS` batch reads of size `READ_SIZE` and
/// `NUM_WRITES` batch writes of size `WRITE_SIZE`.
pub struct BasicAdapterInterface<
    T,
    PI,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>, PhantomData<PI>);

impl<
        T,
        PI,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[T; READ_SIZE]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = PI;
}

pub struct VecHeapAdapterInterface<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for VecHeapAdapterInterface<
        T,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Reads = [[[T; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type ProcessedInstruction = MinimalInstruction<T>;
}

pub struct VecHeapTwoReadsAdapterInterface<
    T,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for VecHeapTwoReadsAdapterInterface<
        T,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Reads = (
        [[T; READ_SIZE]; BLOCKS_PER_READ1],
        [[T; READ_SIZE]; BLOCKS_PER_READ2],
    );
    type Writes = [[T; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type ProcessedInstruction = MinimalInstruction<T>;
}

/// Similar to `BasicAdapterInterface`, but it flattens the reads and writes into a single flat
/// array for each
pub struct FlatInterface<T, PI, const READ_CELLS: usize, const WRITE_CELLS: usize>(
    PhantomData<T>,
    PhantomData<PI>,
);

impl<T, PI, const READ_CELLS: usize, const WRITE_CELLS: usize> VmAdapterInterface<T>
    for FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>
{
    type Reads = [T; READ_CELLS];
    type Writes = [T; WRITE_CELLS];
    type ProcessedInstruction = PI;
}

/// An interface that is fully determined during runtime. This should **only** be used as a last
/// resort when static compile-time guarantees cannot be made.
#[derive(Serialize, Deserialize)]
pub struct DynAdapterInterface<T>(PhantomData<T>);

impl<T> VmAdapterInterface<T> for DynAdapterInterface<T> {
    /// Any reads can be flattened into a single vector.
    type Reads = DynArray<T>;
    /// Any writes can be flattened into a single vector.
    type Writes = DynArray<T>;
    /// Any processed instruction can be flattened into a single vector.
    type ProcessedInstruction = DynArray<T>;
}

/// Newtype to implement `From`.
#[derive(Clone, Debug, Default)]
pub struct DynArray<T>(pub Vec<T>);

// =================================================================================================
// Definitions of ProcessedInstruction types for use in integration API
// =================================================================================================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

// This ProcessedInstruction is used by rv32_rdwrite
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
}

// This ProcessedInstruction is used by rv32_jalr
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SignedImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
    /// Sign of the immediate (1 if negative, 0 if positive)
    pub imm_sign: T,
}

// =================================================================================================
// Conversions between adapter interfaces
// =================================================================================================

mod conversions {
    use super::*;

    // AdapterAirContext: VecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> VecHeapAdapterInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> VecHeapTwoReadsAdapterInterface
    impl<
            T: Clone,
            const BLOCKS_PER_READ1: usize,
            const BLOCKS_PER_READ2: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            VecHeapTwoReadsAdapterInterface<
                T,
                BLOCKS_PER_READ1,
                BLOCKS_PER_READ2,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: BasicInterface -> VecHeapAdapterInterface
    impl<
            T,
            PI: Into<MinimalInstruction<T>>,
            const BASIC_NUM_READS: usize,
            const BASIC_NUM_WRITES: usize,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        >
        for AdapterAirContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            assert_eq!(BASIC_NUM_READS, NUM_READS * BLOCKS_PER_READ);
            let mut reads_it = ctx.reads.into_iter();
            let reads = from_fn(|_| from_fn(|_| reads_it.next().unwrap()));
            assert_eq!(BASIC_NUM_WRITES, BLOCKS_PER_WRITE);
            let mut writes_it = ctx.writes.into_iter();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads,
                writes,
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: FlatInterface -> BasicInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>
    {
        /// ## Panics
        /// If `READ_CELLS != NUM_READS * READ_SIZE` or `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>> {
            assert_eq!(READ_CELLS, NUM_READS * READ_SIZE);
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut reads_it = ctx.reads.into_iter().flatten();
            let reads = from_fn(|_| reads_it.next().unwrap());
            let mut writes_it = ctx.writes.into_iter().flatten();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads,
                writes,
                instruction: ctx.instruction,
            }
        }
    }

    // AdapterAirContext: BasicInterface -> FlatInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        > From<AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    {
        /// ## Panics
        /// If `READ_CELLS != NUM_READS * READ_SIZE` or `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            AdapterAirContext {
                to_pc,
                reads,
                writes,
                instruction,
            }: AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>,
        ) -> AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        > {
            assert_eq!(READ_CELLS, NUM_READS * READ_SIZE);
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut reads_it = reads.into_iter();
            let reads: [[T; READ_SIZE]; NUM_READS] =
                from_fn(|_| from_fn(|_| reads_it.next().unwrap()));
            let mut writes_it = writes.into_iter();
            let writes: [[T; WRITE_SIZE]; NUM_WRITES] =
                from_fn(|_| from_fn(|_| writes_it.next().unwrap()));
            AdapterAirContext {
                to_pc,
                reads,
                writes,
                instruction,
            }
        }
    }

    impl<T> From<Vec<T>> for DynArray<T> {
        fn from(v: Vec<T>) -> Self {
            Self(v)
        }
    }

    impl<T> From<DynArray<T>> for Vec<T> {
        fn from(v: DynArray<T>) -> Vec<T> {
            v.0
        }
    }

    impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for DynArray<T> {
        fn from(v: [[T; N]; M]) -> Self {
            Self(v.into_iter().flatten().collect())
        }
    }

    impl<T, const N: usize, const M: usize> From<DynArray<T>> for [[T; N]; M] {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(v.0.len(), N * M, "Incorrect vector length {}", v.0.len());
            let mut it = v.0.into_iter();
            from_fn(|_| from_fn(|_| it.next().unwrap()))
        }
    }

    impl<T, const N: usize, const M: usize, const R: usize> From<[[[T; N]; M]; R]> for DynArray<T> {
        fn from(v: [[[T; N]; M]; R]) -> Self {
            Self(
                v.into_iter()
                    .flat_map(|x| x.into_iter().flatten())
                    .collect(),
            )
        }
    }

    impl<T, const N: usize, const M: usize, const R: usize> From<DynArray<T>> for [[[T; N]; M]; R] {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(
                v.0.len(),
                N * M * R,
                "Incorrect vector length {}",
                v.0.len()
            );
            let mut it = v.0.into_iter();
            from_fn(|_| from_fn(|_| from_fn(|_| it.next().unwrap())))
        }
    }

    impl<T, const N: usize, const M1: usize, const M2: usize> From<([[T; N]; M1], [[T; N]; M2])>
        for DynArray<T>
    {
        fn from(v: ([[T; N]; M1], [[T; N]; M2])) -> Self {
            let vec =
                v.0.into_iter()
                    .flatten()
                    .chain(v.1.into_iter().flatten())
                    .collect();
            Self(vec)
        }
    }

    impl<T, const N: usize, const M1: usize, const M2: usize> From<DynArray<T>>
        for ([[T; N]; M1], [[T; N]; M2])
    {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(
                v.0.len(),
                N * (M1 + M2),
                "Incorrect vector length {}",
                v.0.len()
            );
            let mut it = v.0.into_iter();
            (
                from_fn(|_| from_fn(|_| it.next().unwrap())),
                from_fn(|_| from_fn(|_| it.next().unwrap())),
            )
        }
    }

    // AdapterAirContext: BasicInterface -> DynInterface
    impl<
            T,
            PI: Into<DynArray<T>>,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> BasicInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    where
        PI: From<DynArray<T>>,
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: FlatInterface -> DynInterface
    impl<T: Clone, PI: Into<DynArray<T>>, const READ_CELLS: usize, const WRITE_CELLS: usize>
        From<AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(ctx: AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.to_vec().into(),
                writes: ctx.writes.to_vec().into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    impl<T> From<MinimalInstruction<T>> for DynArray<T> {
        fn from(m: MinimalInstruction<T>) -> Self {
            Self(vec![m.is_valid, m.opcode])
        }
    }

    impl<T> From<DynArray<T>> for MinimalInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            MinimalInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
            }
        }
    }

    impl<T> From<DynArray<T>> for ImmInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            ImmInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
                immediate: m.next().unwrap(),
            }
        }
    }

    impl<T> From<ImmInstruction<T>> for DynArray<T> {
        fn from(instruction: ImmInstruction<T>) -> Self {
            DynArray::from(vec![
                instruction.is_valid,
                instruction.opcode,
                instruction.immediate,
            ])
        }
    }
}
