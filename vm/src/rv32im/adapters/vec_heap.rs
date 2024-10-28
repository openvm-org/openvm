use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    iter::{once, zip},
    marker::PhantomData,
};

use ax_circuit_derive::AlignedBorrow;
use ax_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::instruction::Instruction;
use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{read_rv32_register, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VecHeapAdapterInterface, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};

/// This adapter reads from R (R <= 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers
///   are read from registers (address space 1).
/// * Reads take the form of `NUM_READS` consecutive reads of size `READ_SIZE`
///   from the heap, starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `NUM_WRITES` consecutive writes of size `WRITE_SIZE`
///   to the heap, starting from the address in `rd`.
#[derive(Debug)]
pub struct Rv32VecHeapAdapterChip<
    F: Field,
    const R: usize,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: Rv32VecHeapAdapterAir<R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
    _marker: PhantomData<F>,
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
        assert!(R <= 2);
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let address_bits = memory_controller.mem_config.pointer_max_bits;
        Self {
            air: Rv32VecHeapAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                address_bits,
            },
            _marker: PhantomData,
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
    pub rs: [MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>; R],
    /// Read register value from address space d=1
    pub rd: MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>,

    pub rd_val: F,

    pub reads: [[MemoryReadRecord<F, READ_SIZE>; NUM_READS]; R],
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

    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; R],

    pub rs_read_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>; R],
    pub rd_read_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>,

    pub reads_aux: [[MemoryReadAuxCols<T, READ_SIZE>; NUM_READS]; R],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
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
        VecHeapAdapterInterface<AB::Expr, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

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
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
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
        let mut reg_val_f: Vec<_> = cols
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
        let rd_val_f = reg_val_f.pop().unwrap();
        let rs_val_f = reg_val_f;

        let e = AB::F::from_canonical_usize(2);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux,) {
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
                        rd_val_f.clone() + AB::Expr::from_canonical_usize(i * WRITE_SIZE),
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
                    cols.rs_ptr
                        .first()
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::zero()),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::zero()),
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
    type Interface = VecHeapAdapterInterface<F, R, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

        let mut rs_vals = [0; R];
        let rs_records: [_; R] = from_fn(|i| {
            let addr = if i == 0 { b } else { c };
            let (record, val) = read_rv32_register(memory, d, addr);
            rs_vals[i] = val;
            record
        });
        let (rd_record, rd_val) = read_rv32_register(memory, d, a);

        let read_records = rs_vals.map(|address| {
            // TODO: assert address has < 2^address_bits
            from_fn(|i| {
                memory.read::<READ_SIZE>(e, F::from_canonical_u32(address + (i * READ_SIZE) as u32))
            })
        });
        let read_data = read_records.map(|r| r.map(|x| x.data));

        let record = Rv32VecHeapReadRecord {
            rs: rs_records,
            rd: rd_record,
            rd_val: F::from_canonical_u32(rd_val),
            reads: read_records,
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
        aux_cols_factory: &MemoryAuxColsFactory<F>,
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
            .map(|r| aux_cols_factory.make_read_aux_cols(r));
        row_slice.rd_read_aux = aux_cols_factory.make_read_aux_cols(read_record.rd);
        row_slice.reads_aux = read_record
            .reads
            .map(|r| r.map(|x| aux_cols_factory.make_read_aux_cols(x)));
        row_slice.writes_aux = write_record
            .writes
            .map(|w| aux_cols_factory.make_write_aux_cols(w));
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
