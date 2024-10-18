use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    iter::zip,
    marker::PhantomData,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

/// This adapter reads from 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers
///   are read from registers (address space 1).
/// * Reads take the form of `NUM_READS` consecutive reads of size `READ_SIZE`
///   from the heap, starting from the addresses in `rs1` and `rs2`.
/// * Writes take the form of `NUM_WRITES` consecutive writes of size `WRITE_SIZE`
///   to the heap, starting from the address in `rd`.
#[derive(Clone, Debug)]
pub struct NativeVecHeapAdapterChip<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: NativeVecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > NativeVecHeapAdapterChip<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        let address_bits = memory_controller.mem_config.pointer_max_bits;
        Self {
            air: NativeVecHeapAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                address_bits,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeVecHeapReadRecord<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    /// Read register value from address space e=1
    pub rs1: MemoryReadRecord<F, 1>,
    /// Read register value from address space op_f=1
    pub rs2: MemoryReadRecord<F, 1>,
    /// Read register value from address space d=1
    pub rd: MemoryReadRecord<F, 1>,

    pub rd_val: F,

    pub ptr_as: F,
    pub heap_as: F,

    pub reads1: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
    pub reads2: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
}

#[derive(Clone, Debug)]
pub struct NativeVecHeapWriteRecord<F: Field, const NUM_WRITES: usize, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<u32>,

    pub writes: [MemoryWriteRecord<F, WRITE_SIZE>; NUM_WRITES],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeVecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,

    pub ptr_as: T,
    pub heap_as: T,

    pub rd_val: T,
    pub rs1_val: T,
    pub rs2_val: T,

    pub rs1_read_aux: MemoryReadAuxCols<T, 1>,
    pub rs2_read_aux: MemoryReadAuxCols<T, 1>,
    pub rd_read_aux: MemoryReadAuxCols<T, 1>,

    pub reads1_aux: [MemoryReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub reads2_aux: [MemoryReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
}

#[derive(Clone)]
pub struct NativeVecHeapAdapterInterface<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[[T; READ_SIZE]; NUM_READS]; 2];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = MinimalInstruction<T>;
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeVecHeapAdapterAir<
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F> for NativeVecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        NativeVecHeapAdapterCols::<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB> for NativeVecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Interface =
        NativeVecHeapAdapterInterface<AB::Expr, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeVecHeapAdapterCols<_, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Read register values for rs1, rs2, rd
        for (ptr, val, aux) in [
            (cols.rs1_ptr, cols.rs1_val, &cols.rs1_read_aux),
            (cols.rs2_ptr, cols.rs2_val, &cols.rs2_read_aux),
            (cols.rd_ptr, cols.rd_val, &cols.rd_read_aux),
        ] {
            self.memory_bridge
                .read(
                    MemoryAddress::new(cols.ptr_as, ptr),
                    [val.into()],
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Reads from heap
        for (address, reads, reads_aux) in izip!(
            [cols.rs1_val, cols.rs2_val],
            ctx.reads,
            [&cols.reads1_aux, &cols.reads2_aux]
        ) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            cols.heap_as,
                            address.clone() + AB::Expr::from_canonical_usize(i * READ_SIZE),
                        ),
                        read,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        cols.heap_as,
                        cols.rd_val.clone() + AB::Expr::from_canonical_usize(i * WRITE_SIZE),
                    ),
                    write,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_pc_custom(
                ctx.instruction.opcode,
                [
                    cols.rd_ptr.into(),
                    cols.rs1_ptr.into(),
                    cols.rs2_ptr.into(),
                    cols.ptr_as.into(),
                    cols.heap_as.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                AB::Expr::from_canonical_u8(4),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeVecHeapAdapterCols<_, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterChip<F>
    for NativeVecHeapAdapterChip<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = NativeVecHeapReadRecord<F, NUM_READS, READ_SIZE>;
    type WriteRecord = NativeVecHeapWriteRecord<F, NUM_WRITES, WRITE_SIZE>;
    type Air = NativeVecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;
    type Interface = NativeVecHeapAdapterInterface<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            ..
        } = *instruction;

        let rs1_record = memory.read_cell(d, b);
        let rs2_record = memory.read_cell(d, c);
        let rd_record = memory.read_cell(d, a);

        let [reads1, reads2] = [rs1_record.data[0], rs2_record.data[0]].map(|address| {
            // TODO: assert address has < 2^address_bits
            from_fn(|i| {
                memory.read::<READ_SIZE>(e, address + F::from_canonical_u32((i * READ_SIZE) as u32))
            })
        });
        let reads = [reads1.map(|x| x.data), reads2.map(|x| x.data)];

        let record = NativeVecHeapReadRecord {
            rs1: rs1_record,
            rs2: rs2_record,
            rd: rd_record,
            rd_val: rd_record.data[0],
            ptr_as: d,
            heap_as: e,
            reads1,
            reads2,
        };

        Ok((reads, record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let e = instruction.e;
        let mut i = 0;
        let writes = output.writes.map(|write| {
            let record = memory.write(
                e,
                read_record.rd_val + F::from_canonical_u32((i * WRITE_SIZE) as u32),
                write,
            );
            i += 1;
            record
        });

        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, writes },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let row_slice: &mut NativeVecHeapAdapterCols<
            F,
            NUM_READS,
            NUM_WRITES,
            READ_SIZE,
            WRITE_SIZE,
        > = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.rd_ptr = read_record.rd.pointer;
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2_ptr = read_record.rs2.pointer;

        row_slice.rd_val = read_record.rd.data[0];
        row_slice.rs1_val = read_record.rs1.data[0];
        row_slice.rs2_val = read_record.rs2.data[0];

        row_slice.ptr_as = read_record.ptr_as;
        row_slice.heap_as = read_record.heap_as;

        row_slice.rs1_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rs1);
        row_slice.rs2_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rs2);
        row_slice.rd_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rd);
        row_slice.reads1_aux = read_record
            .reads1
            .map(|r| self.aux_cols_factory.make_read_aux_cols(r));
        row_slice.reads2_aux = read_record
            .reads2
            .map(|r| self.aux_cols_factory.make_read_aux_cols(r));
        row_slice.writes_aux = write_record
            .writes
            .map(|w| self.aux_cols_factory.make_write_aux_cols(w));
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

mod conversions {
    use super::NativeVecHeapAdapterInterface;
    use crate::arch::{AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface};

    // AdapterAirContext: NativeVecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
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

    // AdapterRuntimeContext: NativeVecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterRuntimeContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> NativeVecHeapAdapterInterface
    impl<
            T,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
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

    // AdapterRuntimeContext: DynInterface -> NativeVecHeapAdapterInterface
    impl<
            T,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterRuntimeContext<T, DynAdapterInterface<T>>>
        for AdapterRuntimeContext<
            T,
            NativeVecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    {
        fn from(ctx: AdapterRuntimeContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }
}
