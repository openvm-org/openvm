# Execution records

## Motivation

At the end of the [preflight execution](vm.md#preflight-execution), each chip needs to generate its trace. However, only some part of the trace is usually sufficient to determine the rest. Therefore, during the instruction execution stage, each chip will gather only necessary information about the trace. This way, serial execution is sped up, and in the trace generation stage all chips can restore their traces simultaneously from the stored information.

## Records

For storing this minimal information, each chip defines its own _Record_ type. Most of the times, one instruction results in generating one record, which corresponds to one trace row, but there is no such hard rule, and some chips use different models. In general, how to generate records from an instruction and how to interpret the record afterwards is up to the chip.

Below we discuss how the details of records generation and the required properties of the records, as well as associated types.

### Record arenas

An entity for storing the records is called a _Record arena_. Its one function is to allocate a memory slice for a new record. During executing an instruction in preflight execution, a chip will request the memory slice to write its record to from a record arena, and then fill it. Here is an example:

```rust
fn execute(
    &self,
    state: VmStateMut<F, TracingMemory, RA>,
    instruction: &Instruction<F>,
) -> Result<(), ExecutionError> {
    let record: &mut PhantomRecord = state.ctx.alloc(EmptyMultiRowLayout::default());
    let pc = *state.pc;
    record.pc = pc;
    record.timestamp = state.memory.timestamp;
    let [a, b, c] = [instruction.a, instruction.b, instruction.c].map(|x| x.as_canonical_u32());
    record.operands = [a, b, c];
    // ...
}
```

> [!NOTE]
> One can see from this example that a chip does not own its record arena, which is provided to it with the state instead.

### Types of record arenas

We have two implementations of record arenas: `MatrixRecordArena` for generating trace on CPU, and `DenseRecordArena` for GPU. The distinction is due to the fact that, for GPU tracegen, we have to first send the records to the GPU, and then generate the trace on device. This motivates the arena to pack the record densely, hence the name. On the other hand, if we don't need to send the records anywhere, then they may already represent a partially filled trace matrix -- maybe with gaps between their elements, which will be filled during tracegen.

### `RecordMut`

We said that the record arena returns a mutable memory slice, which the chip will interpret as a record. However, this is not technically correct, because the chip, in fact, interprets it as a _mutable record view_. We indicate this distinction by the fact that the `RecordArena` trait depends on the `RecordMut` type, which it returns on allocation. In most cases, but not always, `RecordMut = &mut Record`.

```rust
pub trait RecordArena<'a, Layout, RecordMut> {
    fn alloc(&'a mut self, layout: Layout) -> RecordMut;
}
```

We may refer to this mutable record view as simply "record" for brewity.

### Layout and Metadata

A record arena needs to know how much memory to allocate. A struct with this information is called _Layout_.

In most cases, all records for the chip have constant size, in which case the record type already uniquely defines its size. In other cases, a layout type is usually a struct that contains a _metadata_, which the layout type interprets in its way to define the required size.

More specifically, there is a trait `SizedRecord` that, given the layout, decides the required record size and alignment.

```rust
pub trait SizedRecord<Layout> {
    fn size(layout: &Layout) -> usize;
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
```

#### `AdapterCoreLayout`

When a chip consists of a core and an adapter, it makes sense that a record consists of "an adapter part" and "a core part". For such purposes we have the `AdapterCoreLayout` struct:

```rust
pub struct AdapterCoreLayout<M> {
    pub metadata: M,
}
```

Both adapter and core record types (call them `A` and `C`) would use the same layout, and the overall record type in this case is `(A, C)`:

```rust
impl<'a, F: Field, A, C, M: AdapterCoreMetadata> RecordArena<'a, AdapterCoreLayout<M>, (A, C)>
    for MatrixRecordArena<F>
where
    [u8]: CustomBorrow<'a, A, AdapterCoreLayout<M>> + CustomBorrow<'a, C, AdapterCoreLayout<M>>,
    M: Clone,
{
    fn alloc(&'a mut self, layout: AdapterCoreLayout<M>) -> (A, C) {
        let adapter_width = M::get_adapter_width();
        let buffer = self.alloc_single_row();
        let (adapter_buffer, core_buffer) = unsafe { buffer.split_at_mut_unchecked(adapter_width) };

        let adapter_record: A = adapter_buffer.custom_borrow(layout.clone());
        let core_record: C = core_buffer.custom_borrow(layout);

        (adapter_record, core_record)
    }
}
```

An example of using such layout type:

```rust
impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA>
    for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData: Into<[[u8; NUM_LIMBS]; 2]>, WriteData = ()>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BranchEqualCoreRecord<NUM_LIMBS>,
        ),
    >,
{
    // ...

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // ...
    }
}
```

#### `MultiRowLayout`

Another example of a layout is a `MultiRowLayout`, which may treat its metadata as the number of trace rows the record will correspond to, or ignore its metadata and assume that one instruction will generate one trace row.

```rust
pub struct MultiRowLayout<M> {
    pub metadata: M,
}

pub trait MultiRowMetadata {
    fn get_num_rows(&self) -> usize;
}

pub struct EmptyMultiRowMetadata {}

impl MultiRowMetadata for EmptyMultiRowMetadata {
    fn get_num_rows(&self) -> usize {
        1
    }
}

pub struct FriReducedOpeningMetadata {
    length: usize,
    is_init: bool,
}

impl MultiRowMetadata for FriReducedOpeningMetadata {
    fn get_num_rows(&self) -> usize {
        self.length + 2
    }
}
```

Finally, `MatrixRecordArena` will directly use the number of rows to find out the record size:

```rust
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
```

### `CustomBorrow`

The last code listing introduced a new `CustomBorrow` trait. It is implemented for `[u8]` and it serves two purposes:

- Given a layout, allocate the required `RecordMut` and return,
- Assuming that the start of the `[u8]` is filled with some record, extract the layout for this record.

The latter is important as it establishes a guarantee that **the layout must be restorable from the record**. In particular, if an arena would just generate some variable number of rows from an instruction to fill later, then, posterior to their filling, the number of rows from this instruction must be defined by the contents of the trace rows:

```rust
impl<'a, F> CustomBorrow<'a, FriReducedOpeningRecordMut<'a, F>, FriReducedOpeningLayout>
    for [u8]
{
    // ... 
    unsafe fn extract_layout(&self) -> FriReducedOpeningLayout {
        let header: &FriReducedOpeningHeaderRecord = self.borrow();
        FriReducedOpeningLayout::new(FriReducedOpeningMetadata {
            length: header.length as usize,
            is_init: header.is_init,
        })
    }
}
```

### `RecordSeeker`

When a chip has to generate trace using the records it has, it needs to extract the records from the record arena, transform each of them into a number of trace rows, and concatenate. To do this for `DenseRecordArena`, we use a special `RecordSeeker` struct:

```rust
pub struct RecordSeeker<'a, RA, RecordMut, Layout> {
    pub buffer: &'a mut [u8],
    _phantom: PhantomData<(RA, RecordMut, Layout)>,
}
```

When the size of each record may vary, the flow of the GPU trace generation would be:
- Iterate over the records, remembering their starting offsets in the arena (and, possibly, something else, such as what part of the trace each individual record corresponds to),
- Delegate those offsets to the GPU cores to extract records at them and generate the trace.

In order to do the iteration, record seekers provide useful methods:

```rust
impl<'a, A, C, M> RecordSeeker<'a, DenseRecordArena, (A, C), AdapterCoreLayout<M>>
where
    [u8]: CustomBorrow<'a, A, AdapterCoreLayout<M>> + CustomBorrow<'a, C, AdapterCoreLayout<M>>,
    A: SizedRecord<AdapterCoreLayout<M>>,
    C: SizedRecord<AdapterCoreLayout<M>>,
    M: AdapterCoreMetadata + Clone,
{
    // Returns the aligned sizes of the adapter and core records given their layout
    pub fn get_aligned_sizes(layout: &AdapterCoreLayout<M>) -> (usize, usize) {
        // ...
    }

    // Returns the aligned size of a single record given its layout
    pub fn get_aligned_record_size(layout: &AdapterCoreLayout<M>) -> usize {
        // ...
    }

    // Returns a record at the given offset in the buffer
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
}
```