use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    iter::{once, zip},
    marker::PhantomData,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{read_rv32_register, RV32_CELL_BITS, RV32_REGISTER_NUM_LANES};
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

/// This adapter reads from R (R = 1 or 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers
///   are read from registers (address space 1).
/// * Reads take the form of `NUM_READS` consecutive reads of size `READ_SIZE`
///   from the heap, starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `NUM_WRITES` consecutive writes of size `WRITE_SIZE`
///   to the heap, starting from the address in `rd`.
#[derive(Clone, Debug)]
pub struct Rv32VecHeapAdapterChip<
    F: Field,
    const R: usize,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: Rv32VecHeapAdapterAir<R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<
        F: PrimeField32,
        const R: usize,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > Rv32VecHeapAdapterChip<F, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        assert!(R == 1 || R == 2);
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        let address_bits = memory_controller.mem_config.pointer_max_bits;
        Self {
            air: Rv32VecHeapAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                address_bits,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32VecHeapReadRecord<
    F: Field,
    const R: usize,
    const NUM_READS: usize,
    const READ_SIZE: usize,
> {
    /// Read register value from address space e=1
    pub rs: [MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>; R],
    /// Read register value from address space op_f=1
    // pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Read register value from address space d=1
    pub rd: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,

    pub rd_val: F,

    pub reads: [[MemoryReadRecord<F, READ_SIZE>; NUM_READS]; R],
    // pub reads2: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
}

#[derive(Clone, Debug)]
pub struct Rv32VecHeapWriteRecord<F: Field, const NUM_WRITES: usize, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<u32>,

    pub writes: [MemoryWriteRecord<F, WRITE_SIZE>; NUM_WRITES],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32VecHeapAdapterCols<
    T,
    const R: usize,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rd_ptr: T,
    pub rs_ptr: [T; R],

    pub rd_val: [T; RV32_REGISTER_NUM_LANES],
    pub rs_val: [[T; RV32_REGISTER_NUM_LANES]; R],

    pub rs_read_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; R],
    pub rd_read_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,

    pub reads_aux: [[MemoryReadAuxCols<T, READ_SIZE>; NUM_READS]; R],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
}

#[derive(Clone)]
pub struct Rv32VecHeapAdapterInterface<
    T,
    const R: usize,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const R: usize,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[[T; READ_SIZE]; NUM_READS]; R];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = MinimalInstruction<T>;
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32VecHeapAdapterAir<
    const R: usize,
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
        const R: usize,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F> for Rv32VecHeapAdapterAir<R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        Rv32VecHeapAdapterCols::<F, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const R: usize,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB> for Rv32VecHeapAdapterAir<R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Interface =
        Rv32VecHeapAdapterInterface<AB::Expr, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32VecHeapAdapterCols<_, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd
        let read_reg = cols
            .rs_ptr
            .into_iter()
            .zip(cols.rs_val)
            .zip(cols.rs_read_aux.iter())
            .map(|((a, b), c)| (a, b, c))
            .chain(once((cols.rd_ptr, cols.rd_val, &cols.rd_read_aux)));
        for (ptr, val, aux) in read_reg {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::one(), ptr),
                    val,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with
        // a range check on the highest limb.
        let register_val_f: Vec<_> = cols
            .rs_val
            .iter()
            .chain(once(&cols.rd_val))
            .map(|&decomp| {
                // TODO: range check
                decomp
                    .into_iter()
                    .enumerate()
                    .fold(AB::Expr::zero(), |acc, (i, limb)| {
                        acc + limb * AB::Expr::from_canonical_usize(1 << (i * RV32_CELL_BITS))
                    })
            })
            .collect();

        let e = AB::F::from_canonical_usize(2);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(
            register_val_f[..R].iter(),
            // [rs1_val_f, rs2_val_f],
            ctx.reads,
            &cols.reads_aux,
        ) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
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
                        e,
                        register_val_f[R].clone() + AB::Expr::from_canonical_usize(i * WRITE_SIZE),
                    ),
                    write,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rd_ptr.into(),
                    cols.rs_ptr[0].into(),
                    if R >= 2 {
                        cols.rs_ptr[1].into()
                    } else {
                        AB::Expr::zero()
                    },
                    AB::Expr::one(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32VecHeapAdapterCols<_, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

impl<
        F: PrimeField32,
        const R: usize,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterChip<F>
    for Rv32VecHeapAdapterChip<F, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = Rv32VecHeapReadRecord<F, R, NUM_READS, READ_SIZE>;
    type WriteRecord = Rv32VecHeapWriteRecord<F, NUM_WRITES, WRITE_SIZE>;
    type Air = Rv32VecHeapAdapterAir<R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;
    type Interface =
        Rv32VecHeapAdapterInterface<F, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

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

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

        let (rs1_record, rs1_val) = read_rv32_register(memory, d, b);
        let mut register_records = [rs1_record; R];
        let data1 = from_fn(|i| {
            memory.read::<READ_SIZE>(e, F::from_canonical_u32(rs1_val + (i * READ_SIZE) as u32))
        });
        let mut read_record = [data1; R];
        let mut read_data = [data1.map(|x| x.data); R];
        if R == 2 {
            println!("yo!! read second input");
            let (rs2_record, rs2_val) = read_rv32_register(memory, d, c);
            register_records[1] = rs2_record;
            let data2 = from_fn(|i| {
                memory.read::<READ_SIZE>(e, F::from_canonical_u32(rs2_val + (i * READ_SIZE) as u32))
            });
            read_record[1] = data2;
            read_data[1] = data2.map(|x| x.data);
        }
        let (rd_record, rd_val) = read_rv32_register(memory, d, a);

        let record = Rv32VecHeapReadRecord {
            rs: register_records,
            rd: rd_record,
            rd_val: F::from_canonical_u32(rd_val),
            reads: read_record,
        };

        Ok((read_data, record))
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
        let row_slice: &mut Rv32VecHeapAdapterCols<
            F,
            R,
            NUM_READS,
            NUM_WRITES,
            READ_SIZE,
            WRITE_SIZE,
        > = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.rd_ptr = read_record.rd.pointer;
        row_slice.rs_ptr = read_record.rs.map(|r| r.pointer);

        row_slice.rd_val = read_record.rd.data;
        row_slice.rs_val = read_record.rs.map(|r| r.data);

        row_slice.rs_read_aux = read_record
            .rs
            .map(|r| self.aux_cols_factory.make_read_aux_cols(r));
        row_slice.rd_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rd);
        row_slice.reads_aux = read_record
            .reads
            .map(|r| r.map(|x| self.aux_cols_factory.make_read_aux_cols(x)));
        row_slice.writes_aux = write_record
            .writes
            .map(|w| self.aux_cols_factory.make_write_aux_cols(w));
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

mod conversions {
    use super::Rv32VecHeapAdapterInterface;
    use crate::arch::{AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface};

    // AdapterAirContext: Rv32VecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const R: usize,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
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

    // AdapterRuntimeContext: Rv32VecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const R: usize,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterRuntimeContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> Rv32VecHeapAdapterInterface
    impl<
            T,
            const R: usize,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
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

    // AdapterRuntimeContext: DynInterface -> Rv32VecHeapAdapterInterface
    impl<
            T,
            const R: usize,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterRuntimeContext<T, DynAdapterInterface<T>>>
        for AdapterRuntimeContext<
            T,
            Rv32VecHeapAdapterInterface<T, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
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
