use std::{
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

use super::{read_rv32_register, RV32_CELL_BITS, RV32_REGISTER_NUM_LANES};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
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
pub struct Rv32VecHeapAdapterChip<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > Rv32VecHeapAdapterChip<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: Rv32VecHeapAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32VecHeapReadRecord<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    /// Read register value from address space e=1
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Read register value from address space op_f=1
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Read register value from address space d=1
    pub rd: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,

    pub rd_val: F,

    pub reads1: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
    pub reads2: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
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
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,

    pub rd_val: [T; RV32_REGISTER_NUM_LANES],
    pub rs1_val: [T; RV32_REGISTER_NUM_LANES],
    pub rs2_val: [T; RV32_REGISTER_NUM_LANES],

    pub rs1_read_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub rs2_read_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub rd_read_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,

    pub reads1_aux: [MemoryReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub reads2_aux: [MemoryReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
}

#[derive(Clone)]
struct Rv32VecHeapAdapterReads<T, const NUM_READS: usize, const READ_SIZE: usize> {
    pub data: [[[T; READ_SIZE]; NUM_READS]; 2],
}

#[derive(Clone)]
pub struct Rv32VecHeapAdapterInterface<
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
    for Rv32VecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = Rv32VecHeapAdapterReads<T, NUM_READS, READ_SIZE>;
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = MinimalInstruction<T>;
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32VecHeapAdapterAir<
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
    > BaseAir<F> for Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        Rv32VecHeapAdapterCols::<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB> for Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Interface =
        Rv32VecHeapAdapterInterface<AB::Expr, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32VecHeapAdapterCols<_, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Read register values for rs1, rs2, rd
        for (ptr, val, aux) in [
            (cols.rs1_ptr, cols.rs1_val, cols.rs1_read_aux),
            (cols.rs2_ptr, cols.rs2_val, cols.rs2_read_aux),
            (cols.rd_ptr, cols.rd_val, cols.rd_read_aux),
        ] {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::one(), ptr),
                    val,
                    timestamp_pp(),
                    &cols.rs1_read_aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with
        // a range check on the highest limb.
        let [rs1_val_f, rs2_val_f, rd_val_f] =
            [cols.rs1_val, cols.rs2_val, cols.rd_val].map(|decomp| {
                decomp
                    .into_iter()
                    .enumerate()
                    .fold(AB::Expr::zero(), |acc, (i, limb)| {
                        acc + limb * AB::Expr::from_canonical_usize(1 << (i * RV32_CELL_BITS))
                    })
            });

        let e = AB::F::from_canonical_usize(2);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(
            [rs1_val_f, rs2_val_f],
            ctx.reads.data,
            [&cols.reads1_aux, &cols.reads2_aux]
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
                        &aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_usize(2),
                        rd_val_f.clone() + AB::Expr::from_canonical_usize(i * WRITE_SIZE),
                    ),
                    write,
                    timestamp_pp(),
                    &aux,
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
                    AB::Expr::one(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                AB::Expr::from_canonical_u8(4),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32VecHeapAdapterCols<_, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE> =
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
    > VmAdapterChip<F> for Rv32VecHeapAdapterChip<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = Rv32VecHeapReadRecord<F, NUM_READS, READ_SIZE>;
    type WriteRecord = Rv32VecHeapWriteRecord<F, NUM_WRITES, WRITE_SIZE>;
    type Air = Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;
    type Interface = BasicAdapterInterface<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

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
            op_f,
            ..
        } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 1);
        debug_assert_eq!(op_f.as_canonical_u32(), 1);

        let rs1 = read_rv32_register(memory, d, b);
        let rs2 = read_rv32_register(memory, e, c);
        let rd = read_rv32_register(memory, op_f, a);

        let reads1 = core::array::from_fn(|i| {
            memory.read::<READ_SIZE>(
                F::from_canonical_u32(2),
                F::from_canonical_u32(rs1.1 + (i * READ_SIZE) as u32),
            )
        });

        let reads2 = core::array::from_fn(|i| {
            memory.read::<READ_SIZE>(
                F::from_canonical_u32(2),
                F::from_canonical_u32(rs2.1 + (i * READ_SIZE) as u32),
            )
        });

        let reads = Rv32VecHeapAdapterReads {
            rs1: rs1.0.data,
            rs2: rs2.0.data,
            rd: rd.0.data,
            reads: [reads1.map(|x| x.data), reads2.map(|x| x.data)],
        };

        let record = Rv32VecHeapReadRecord {
            rs1: rs1.0,
            rs2: rs2.0,
            rd: rd.0,
            rd_val: F::from_canonical_u32(rd.1),
            reads1,
            reads2,
        };

        Ok((reads, record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let writes = core::array::from_fn(|i| {
            memory.write(
                F::from_canonical_u32(2),
                read_record.rd_val + F::from_canonical_u32((i * WRITE_SIZE) as u32),
                output.writes[i],
            )
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
            NUM_READS,
            NUM_WRITES,
            READ_SIZE,
            WRITE_SIZE,
        > = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.rd_ptr = read_record.rd.pointer;
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2_ptr = read_record.rs2.pointer;

        row_slice.rs1_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rs1);
        row_slice.rs2_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rs2);
        row_slice.rd_read_aux = self.aux_cols_factory.make_read_aux_cols(read_record.rd);
        row_slice.reads1_aux = core::array::from_fn(|i| {
            self.aux_cols_factory
                .make_read_aux_cols(read_record.reads1[i])
        });
        row_slice.reads2_aux = core::array::from_fn(|i| {
            self.aux_cols_factory
                .make_read_aux_cols(read_record.reads2[i])
        });
        row_slice.writes_aux = core::array::from_fn(|i| {
            self.aux_cols_factory
                .make_write_aux_cols(write_record.writes[i])
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
