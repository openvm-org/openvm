use std::{
    borrow::BorrowMut,
    io::Cursor,
    marker::PhantomData,
    ptr::{copy_nonoverlapping, slice_from_raw_parts_mut},
};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
};

pub trait Arena {
    /// Currently `width` always refers to the main trace width.
    fn with_capacity(height: usize, width: usize) -> Self;

    fn is_empty(&self) -> bool;

    /// Only used for metric collection purposes. Intended usage is that for a record arena that
    /// corresponds to a single trace matrix, this function can extract the current number of used
    /// rows of the corresponding trace matrix. This is currently expected to work only for
    /// [MatrixRecordArena].
    #[cfg(feature = "metrics")]
    fn current_trace_height(&self) -> usize {
        0
    }
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

/// Helper trait for arenas backed by row-major matrices.
pub trait RowMajorMatrixArena<F>: Arena {
    /// Set the arena's capacity based on the projected trace height.
    fn set_capacity(&mut self, trace_height: usize);
    fn width(&self) -> usize;
    fn trace_offset(&self) -> usize;
    fn into_matrix(self) -> RowMajorMatrix<F>;
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

// =================== Arena Implementations =========================

#[derive(Default)]
pub struct MatrixRecordArena<F> {
    pub trace_buffer: Vec<F>,
    pub width: usize,
    pub trace_offset: usize,
    /// The arena is created with a specified capacity, but may be truncated before being converted
    /// into a [RowMajorMatrix] if `allow_truncate == true`. If `allow_truncate == false`, then the
    /// matrix will never be truncated. The latter is used if the trace matrix must have fixed
    /// dimensions (e.g., for a static verifier).
    pub(super) allow_truncate: bool,
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

    pub fn force_matrix_dimensions(&mut self) {
        self.allow_truncate = false;
    }
}

impl<F: Field> Arena for MatrixRecordArena<F> {
    fn with_capacity(height: usize, width: usize) -> Self {
        let height = next_power_of_two_or_zero(height);
        let trace_buffer = F::zero_vec(height * width);
        Self {
            trace_buffer,
            width,
            trace_offset: 0,
            allow_truncate: true,
        }
    }

    fn is_empty(&self) -> bool {
        self.trace_offset == 0
    }

    #[cfg(feature = "metrics")]
    fn current_trace_height(&self) -> usize {
        self.trace_offset / self.width
    }
}

impl<F: Field> RowMajorMatrixArena<F> for MatrixRecordArena<F> {
    fn set_capacity(&mut self, trace_height: usize) {
        let size = trace_height * self.width;
        // PERF: use memset
        self.trace_buffer.resize(size, F::ZERO);
    }

    fn width(&self) -> usize {
        self.width
    }

    fn trace_offset(&self) -> usize {
        self.trace_offset
    }

    fn into_matrix(mut self) -> RowMajorMatrix<F> {
        let width = self.width();
        assert_eq!(self.trace_offset() % width, 0);
        let rows_used = self.trace_offset() / width;
        let height = next_power_of_two_or_zero(rows_used);
        // This should be automatic since trace_buffer's height is a power of two:
        assert!(height.checked_mul(width).unwrap() <= self.trace_buffer.len());
        if self.allow_truncate {
            self.trace_buffer.truncate(height * width);
        } else {
            assert_eq!(self.trace_buffer.len() % width, 0);
            let height = self.trace_buffer.len() / width;
            assert!(height.is_power_of_two() || height == 0);
        }
        RowMajorMatrix::new(self.trace_buffer, self.width)
    }
}

pub struct DenseRecordArena {
    pub records_buffer: Cursor<Vec<u8>>,
}

const MAX_ALIGNMENT: usize = 32;

impl DenseRecordArena {
    /// Creates a new [DenseRecordArena] with the given capacity in bytes.
    pub fn with_byte_capacity(size_bytes: usize) -> Self {
        let buffer = vec![0; size_bytes + MAX_ALIGNMENT];
        let offset = (MAX_ALIGNMENT - (buffer.as_ptr() as usize % MAX_ALIGNMENT)) % MAX_ALIGNMENT;
        let mut cursor = Cursor::new(buffer);
        cursor.set_position(offset as u64);
        Self {
            records_buffer: cursor,
        }
    }

    pub fn set_byte_capacity(&mut self, size_bytes: usize) {
        let buffer = vec![0; size_bytes + MAX_ALIGNMENT];
        let offset = (MAX_ALIGNMENT - (buffer.as_ptr() as usize % MAX_ALIGNMENT)) % MAX_ALIGNMENT;
        let mut cursor = Cursor::new(buffer);
        cursor.set_position(offset as u64);
        self.records_buffer = cursor;
    }

    /// Returns the allocated size of the arena in bytes.
    ///
    /// **Note**: This may include additional bytes for alignment.
    pub fn capacity(&self) -> usize {
        self.records_buffer.get_ref().len()
    }

    /// Allocates `count` bytes and returns as a mutable slice.
    pub fn alloc_bytes<'a>(&mut self, count: usize) -> &'a mut [u8] {
        let begin = self.records_buffer.position();
        debug_assert!(
            begin as usize + count <= self.records_buffer.get_ref().len(),
            "failed to allocate {count} bytes from {begin} when the capacity is {}",
            self.records_buffer.get_ref().len()
        );
        self.records_buffer.set_position(begin + count as u64);
        // SAFETY:
        // - `begin` is within bounds and caller must ensure `count` bytes are available
        // - The resulting slice is valid for the lifetime of self
        unsafe {
            std::slice::from_raw_parts_mut(
                self.records_buffer
                    .get_mut()
                    .as_mut_ptr()
                    .add(begin as usize),
                count,
            )
        }
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
    pub fn get_record_seeker<R, L>(&mut self) -> RecordSeeker<DenseRecordArena, R, L> {
        RecordSeeker::new(self.allocated_mut())
    }
}

impl Arena for DenseRecordArena {
    // TODO[jpw]: treat `width` as AIR width in number of columns for now
    fn with_capacity(height: usize, width: usize) -> Self {
        let size_bytes = height * (width * size_of::<u32>());
        Self::with_byte_capacity(size_bytes)
    }

    fn is_empty(&self) -> bool {
        self.allocated().is_empty()
    }
}

// =================== Helper Functions =================================

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
        // SAFETY: buffer points to the start of a valid record with proper layout information
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
                // SAFETY:
                // - buff.as_mut_ptr() is valid for len bytes
                // - len matches original buffer size
                // - Bypasses borrow checker for multiple mutable accesses within loop
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
            // SAFETY: offset < len, pointer within buffer bounds
            let src_ptr = unsafe { self.buffer.as_ptr().add(offset) };
            let dst_ptr = arena
                .alloc_buffer(layout.metadata.get_num_rows())
                .as_mut_ptr();
            // SAFETY:
            // - src_ptr points to valid memory with at least aligned_record_size bytes
            // - dst_ptr points to freshly allocated memory with sufficient size
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
    // Returns the aligned sizes of the adapter and core records given their layout
    pub fn get_aligned_sizes(layout: &AdapterCoreLayout<M>) -> (usize, usize) {
        let adapter_alignment = A::alignment(layout);
        let core_alignment = C::alignment(layout);
        let adapter_size = A::size(layout);
        let aligned_adapter_size = adapter_size.next_multiple_of(core_alignment);
        let core_size = C::size(layout);
        let aligned_core_size = (aligned_adapter_size + core_size)
            .next_multiple_of(adapter_alignment)
            - aligned_adapter_size;
        (aligned_adapter_size, aligned_core_size)
    }

    // Returns the aligned size of a single record given its layout
    pub fn get_aligned_record_size(layout: &AdapterCoreLayout<M>) -> usize {
        let (adapter_size, core_size) = Self::get_aligned_sizes(layout);
        adapter_size + core_size
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
        // SAFETY:
        // - adapter_size is calculated to be within the buffer bounds
        // - The buffer has sufficient size for both adapter and core records
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
                // SAFETY:
                // - buff.as_mut_ptr() is valid for len bytes
                // - len matches original buffer size
                // - Bypasses borrow checker for multiple mutable accesses within loop
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
            // SAFETY:
            // - dst_buffer has sufficient size (allocated for a full row)
            // - M::get_adapter_width() is within bounds of the allocated buffer
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

// ============================== MultiRowLayout =======================================

/// Minimal layout information that [RecordArena] requires for record allocation
/// in scenarios involving chips that:
/// - can have multiple rows per record, and
/// - have possibly variable length records
///
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

// ============================== AdapterCoreLayout =======================================
// This is for integration_api usage

/// Minimal layout information that [RecordArena] requires for record allocation
/// in scenarios involving chips that:
/// - have a single row per record, and
/// - have trace row = [adapter_row, core_row]
///
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
/// **NOTE**: `AS` is the adapter type that implements `AdapterTraceExecutor`
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
    AS: super::AdapterTraceExecutor<F>,
{
    #[inline(always)]
    fn get_adapter_width() -> usize {
        AS::WIDTH * size_of::<F>()
    }
}

/// AdapterCoreLayout with empty metadata that can be used by chips that have record type
/// (&mut A, &mut C) where `A` and `C` are `Sized`
pub type EmptyAdapterCoreLayout<F, AS> = AdapterCoreLayout<AdapterCoreEmptyMetadata<F, AS>>;

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
        // SAFETY:
        // - buffer is a freshly allocated row with sufficient size
        // - adapter_width is guaranteed to be less than the total buffer size
        let (adapter_buffer, core_buffer) = unsafe { buffer.split_at_mut_unchecked(adapter_width) };

        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);

        (adapter_record, core_record)
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
        // SAFETY:
        // - buffer has exactly aligned_adapter_size + aligned_core_size bytes
        // - aligned_adapter_size is within bounds by construction
        let (adapter_buffer, core_buffer) =
            unsafe { buffer.split_at_mut_unchecked(aligned_adapter_size) };

        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);

        (adapter_record, core_record)
    }
}
