# Execution records

## Motivation

At the end of the [preflight execution](todo/link), each chip needs to generate its trace. However, only some part of the trace is usually sufficient to determine the rest. Therefore, during the instruction execution stage, each chip will gather only necessary information about the trace. This way, serial execution is sped up, and in the trace generation stage all chips can restore their traces simultaneously from the stored information.

## Records

For storing this minimal information, each chip defines its own _Record_ type. **TODO write that it depends on the backend type, add the code to show this.** Most of the times, one instruction results in generating one record, which corresponds to one trace row, but there is no such hard rule, and some chips use different models. In general, how to generate records from an instruction and how to interpret the record afterwards is up to the chip.

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

### `RecordMut`

We said that the record arena returns a mutable memory slice, which the chip will interpret as a record. However, this is not technically correct, because the chip, in fact, interprets it as a _mutable record view_. We indicate this distinction by the fact that the `RecordArena` trait depends on the `RecordMut` type, which it returns on allocation. In most cases, but not always, `RecordMut = &mut Record`.

```rust
pub trait RecordArena<'a, Layout, RecordMut> {
    fn alloc(&'a mut self, layout: Layout) -> RecordMut;
}
```

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

Another example of a layout is a `MultiRowLayout`, which may treat its metadata as the number of trace rows the record will correpond to, or ignore its metadata and assume that one instruction will generate one trace row.

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