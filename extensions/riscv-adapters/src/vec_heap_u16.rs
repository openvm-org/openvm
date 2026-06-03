use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::{once, zip},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir, BLOCK_FE_WIDTH,
        MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteU16AuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, ptr_to_u16_limbs, tracing_read_reg_ptr,
    tracing_read_u16, tracing_write_u16, RV64_PTR_BITS, RV64_PTR_U16_LIMBS, U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// This adapter reads from R (R <= 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive `BLOCK_FE_WIDTH`-cell reads from the
///   heap, starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive `BLOCK_FE_WIDTH`-cell writes to the
///   heap, starting from the address in `rd`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64VecHeapU16AdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    /// Low 32 bits of rs registers as u16 limbs.
    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    /// Low 32 bits of rd register as u16 limbs.
    pub rd_val: [T; RV64_PTR_U16_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64VecHeapU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>)]
pub struct Rv64VecHeapU16AdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > BaseAir<F> for Rv64VecHeapU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    fn width(&self) -> usize {
        Rv64VecHeapU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > VmAdapterAir<AB> for Rv64VecHeapU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    type Interface = VecHeapAdapterInterface<
        AB::Expr,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        BLOCK_FE_WIDTH,
        BLOCK_FE_WIDTH,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv64VecHeapU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            let bus_payload: [AB::Expr; BLOCK_FE_WIDTH] =
                [val[0].into(), val[1].into(), AB::Expr::ZERO, AB::Expr::ZERO];
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(RV64_REGISTER_AS),
                        byte_ptr_to_u16_ptr::<AB>(ptr),
                    ),
                    bus_payload,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Constrain the high u16 limb of each materialized pointer.
        let limb_shift = AB::F::from_u32(1 << (RV64_PTR_BITS - self.address_bits));
        for val in cols.rs_val.iter().chain(once(&cols.rd_val)) {
            self.range_bus
                .range_check(val[1] * limb_shift, U16_BITS)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the materialized u16 cells of each register value into a memory address.
        let compose = |val: [AB::Var; RV64_PTR_U16_LIMBS]| -> AB::Expr {
            val[0] + val[1] * AB::F::from_u32(1 << U16_BITS)
        };
        let rd_val_f: AB::Expr = compose(cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(compose);

        let e = AB::F::from_u32(RV64_MEMORY_AS);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux,) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                let read_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| read[j].clone());
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
                            byte_ptr_to_u16_ptr::<AB>(
                                address.clone() + AB::Expr::from_usize(i * MEMORY_BLOCK_BYTES),
                            ),
                        ),
                        read_array,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            let write_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| write[j].clone());
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        byte_ptr_to_u16_ptr::<AB>(
                            rd_val_f.clone() + AB::Expr::from_usize(i * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    write_array,
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
                        .unwrap_or(AB::Expr::ZERO),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64VecHeapU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64VecHeapU16AdapterRecord<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptrs: [u32; NUM_READS],
    pub rd_ptr: u32,

    pub rs_vals: [u32; NUM_READS],
    pub rd_val: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub rd_read_aux: MemoryReadAuxRecord,

    pub reads_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv64VecHeapU16AdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64VecHeapU16AdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > AdapterTraceExecutor<F>
    for Rv64VecHeapU16AdapterExecutor<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    const WIDTH: usize =
        Rv64VecHeapU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width();
    type ReadData = [[[u16; BLOCK_FE_WIDTH]; BLOCKS_PER_READ]; NUM_READS];
    type WriteData = [[u16; BLOCK_FE_WIDTH]; BLOCKS_PER_WRITE];
    type RecordMut<'a> =
        &'a mut Rv64VecHeapU16AdapterRecord<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64VecHeapU16AdapterRecord<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>,
    ) -> Self::ReadData {
        let &Instruction { a, b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        // Read register values
        record.rs_vals = from_fn(|i| {
            record.rs_ptrs[i] = if i == 0 { b } else { c }.as_canonical_u32();
            tracing_read_reg_ptr(
                memory,
                record.rs_ptrs[i],
                &mut record.rs_read_aux[i].prev_timestamp,
                self.pointer_max_bits,
            )
        });

        record.rd_ptr = a.as_canonical_u32();
        record.rd_val = tracing_read_reg_ptr(
            memory,
            record.rd_ptr,
            &mut record.rd_read_aux.prev_timestamp,
            self.pointer_max_bits,
        );

        // Read memory values
        from_fn(|i| {
            debug_assert!(
                (record.rs_vals[i] as u64) + ((MEMORY_BLOCK_BYTES * BLOCKS_PER_READ - 1) as u64)
                    < (1u64 << self.pointer_max_bits)
            );
            from_fn(|j| {
                tracing_read_u16(
                    memory,
                    RV64_MEMORY_AS,
                    byte_ptr_to_u16_ptr_value(record.rs_vals[i] + (j * MEMORY_BLOCK_BYTES) as u32),
                    &mut record.reads_aux[i][j].prev_timestamp,
                )
            })
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64VecHeapU16AdapterRecord<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>,
    ) {
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV64_MEMORY_AS);

        debug_assert!(
            (record.rd_val as u64) + ((MEMORY_BLOCK_BYTES * BLOCKS_PER_WRITE - 1) as u64)
                < (1u64 << self.pointer_max_bits)
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..BLOCKS_PER_WRITE {
            tracing_write_u16(
                memory,
                RV64_MEMORY_AS,
                byte_ptr_to_u16_ptr_value(record.rd_val + (i * MEMORY_BLOCK_BYTES) as u32),
                data[i],
                &mut record.writes_aux[i].prev_timestamp,
                &mut record.writes_aux[i].prev_data,
            );
        }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > AdapterTraceFiller<F>
    for Rv64VecHeapU16AdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    const WIDTH: usize =
        Rv64VecHeapU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv64VecHeapU16AdapterRecord<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64VecHeapU16AdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            adapter_row.borrow_mut();

        // Range checks:
        // **NOTE**: Must do the range checks before overwriting the records
        debug_assert!(self.pointer_max_bits <= RV64_PTR_BITS);
        let limb_shift_bits = RV64_PTR_BITS - self.pointer_max_bits;
        for &v in record.rs_vals.iter().chain(once(&record.rd_val)) {
            self.range_checker_chip
                .add_count((v >> U16_BITS) << limb_shift_bits, U16_BITS);
        }

        let timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        record
            .writes_aux
            .iter()
            .rev()
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|(write, cols_write)| {
                let prev: [F; BLOCK_FE_WIDTH] = write.prev_data.map(F::from_u16);
                cols_write.set_prev_data(prev);
                mem_helper.fill(write.prev_timestamp, timestamp_mm(), cols_write.as_mut());
            });

        record
            .reads_aux
            .iter()
            .zip(cols.reads_aux.iter_mut())
            .rev()
            .for_each(|(reads, cols_reads)| {
                reads
                    .iter()
                    .zip(cols_reads.iter_mut())
                    .rev()
                    .for_each(|(read, cols_read)| {
                        mem_helper.fill(read.prev_timestamp, timestamp_mm(), cols_read.as_mut());
                    });
            });

        mem_helper.fill(
            record.rd_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rd_read_aux.as_mut(),
        );

        record
            .rs_read_aux
            .iter()
            .zip(cols.rs_read_aux.iter_mut())
            .rev()
            .for_each(|(aux, cols_aux)| {
                mem_helper.fill(aux.prev_timestamp, timestamp_mm(), cols_aux.as_mut());
            });

        cols.rd_val = ptr_to_u16_limbs(record.rd_val).map(F::from_u16);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = ptr_to_u16_limbs(*val).map(F::from_u16);
            });
        cols.rd_ptr = F::from_u32(record.rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_u32(record.from_timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
