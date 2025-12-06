use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::{once, zip},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, Rv32EcMulAdapterInterface, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::adapters::{
    abstract_compose, tracing_read, tracing_write, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// This adapter reads from 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32EcMulAdapterCols<
    T,
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; 2],
    pub rd_ptr: T,

    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; 2],
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; 2],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_scalar_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_SCALAR],
    pub reads_point_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_POINT],
    pub writes_aux: [MemoryWriteAuxCols<T, POINT_SIZE>; BLOCKS_PER_POINT],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32EcMulAdapterAir<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const BLOCKS_PER_SCALAR: usize,
        const BLOCKS_PER_POINT: usize,
        const SCALAR_SIZE: usize,
        const POINT_SIZE: usize,
    > BaseAir<F>
    for Rv32EcMulAdapterAir<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>
{
    fn width(&self) -> usize {
        Rv32EcMulAdapterCols::<F, BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const BLOCKS_PER_SCALAR: usize,
        const BLOCKS_PER_POINT: usize,
        const SCALAR_SIZE: usize,
        const POINT_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv32EcMulAdapterAir<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>
{
    type Interface = Rv32EcMulAdapterInterface<
        AB::Expr,
        BLOCKS_PER_SCALAR,
        BLOCKS_PER_POINT,
        SCALAR_SIZE,
        POINT_SIZE,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32EcMulAdapterCols<
            _,
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        > = local.borrow();
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
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), ptr),
                    val,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // We constrain the highest limbs of heap pointers to be less than 2^(addr_bits -
        // (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))). This ensures that no overflow
        // occurs when computing memory pointers. Since the number of cells accessed with each
        // address will be small enough, and combined with the memory argument, it ensures
        // that all the cells accessed in the memory are less than 2^addr_bits.
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .chain(std::iter::repeat_n(&cols.rd_val, 2))
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1])
            .collect();

        // range checks constrain to RV32_CELL_BITS bits, so we need to shift the limbs to constrain
        // the correct amount of bits
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        // Note: since limbs are read from memory we already know that limb[i] < 2^RV32_CELL_BITS
        //       thus range checking limb[i] * shift < 2^RV32_CELL_BITS, gives us that
        //       limb[i] < 2^(addr_bits - (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with `abstract_compose`
        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs_val_f: [AB::Expr; 2] = cols.rs_val.map(abstract_compose);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);
        // Reads from heap
        // Scalar reads
        let scalar_address = &rs_val_f[0];
        for (i, (read, aux)) in zip(ctx.reads.0, &cols.reads_scalar_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        scalar_address.clone() + AB::Expr::from_canonical_usize(i * SCALAR_SIZE),
                    ),
                    read,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        // Point reads
        let point_address = &rs_val_f[1];
        for (i, (read, aux)) in zip(ctx.reads.1, &cols.reads_point_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        point_address.clone() + AB::Expr::from_canonical_usize(i * POINT_SIZE),
                    ),
                    read,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        rd_val_f.clone() + AB::Expr::from_canonical_usize(i * POINT_SIZE),
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
                        .unwrap_or(AB::Expr::ZERO),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32EcMulAdapterCols<
            _,
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        > = local.borrow();
        cols.from_state.pc
    }
}

// Intermediate type that should not be copied or cloned and should be directly written to
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32EcMulAdapterRecord<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptrs: [u32; 2],
    pub rd_ptr: u32,

    pub rs_vals: [u32; 2],
    pub rd_val: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; 2],
    pub rd_read_aux: MemoryReadAuxRecord,

    pub reads_scalar_aux: [MemoryReadAuxRecord; BLOCKS_PER_SCALAR],
    pub reads_point_aux: [MemoryReadAuxRecord; BLOCKS_PER_POINT],
    pub writes_aux: [MemoryWriteBytesAuxRecord<POINT_SIZE>; BLOCKS_PER_POINT],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv32EcMulAdapterExecutor<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv32EcMulAdapterFiller<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        F: PrimeField32,
        const BLOCKS_PER_SCALAR: usize,
        const BLOCKS_PER_POINT: usize,
        const SCALAR_SIZE: usize,
        const POINT_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv32EcMulAdapterExecutor<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>
{
    const WIDTH: usize = Rv32EcMulAdapterCols::<
        F,
        BLOCKS_PER_SCALAR,
        BLOCKS_PER_POINT,
        SCALAR_SIZE,
        POINT_SIZE,
    >::width();
    type ReadData = (
        [[u8; SCALAR_SIZE]; BLOCKS_PER_SCALAR],
        [[u8; POINT_SIZE]; BLOCKS_PER_POINT],
    );
    type WriteData = [[u8; POINT_SIZE]; BLOCKS_PER_POINT];
    type RecordMut<'a> = &'a mut Rv32EcMulAdapterRecord<
        BLOCKS_PER_SCALAR,
        BLOCKS_PER_POINT,
        SCALAR_SIZE,
        POINT_SIZE,
    >;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv32EcMulAdapterRecord<
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        >,
    ) -> Self::ReadData {
        let &Instruction { a, b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        record.rs_vals = from_fn(|i| {
            record.rs_ptrs[i] = if i == 0 { b } else { c }.as_canonical_u32();
            u32::from_le_bytes(tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs_ptrs[i],
                &mut record.rs_read_aux[i].prev_timestamp,
            ))
        });

        record.rd_ptr = a.as_canonical_u32();
        record.rd_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            &mut record.rd_read_aux.prev_timestamp,
        ));

        // Read memory values
        debug_assert!(
            (record.rs_vals[0] + (SCALAR_SIZE * BLOCKS_PER_SCALAR - 1) as u32)
                < (1 << self.pointer_max_bits) as u32
        );
        let reads_scalar = from_fn(|i| {
            tracing_read(
                memory,
                RV32_MEMORY_AS,
                record.rs_vals[0] + (i * SCALAR_SIZE) as u32,
                &mut record.reads_scalar_aux[i].prev_timestamp,
            )
        });
        debug_assert!(
            (record.rs_vals[1] + (POINT_SIZE * BLOCKS_PER_POINT - 1) as u32)
                < (1 << self.pointer_max_bits) as u32
        );
        let reads_point = from_fn(|i| {
            tracing_read(
                memory,
                RV32_MEMORY_AS,
                record.rs_vals[1] + (i * POINT_SIZE) as u32,
                &mut record.reads_point_aux[i].prev_timestamp,
            )
        });
        (reads_scalar, reads_point)
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv32EcMulAdapterRecord<
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        >,
    ) {
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV32_MEMORY_AS);

        debug_assert!(
            record.rd_val as usize + POINT_SIZE * BLOCKS_PER_POINT - 1
                < (1 << self.pointer_max_bits)
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..BLOCKS_PER_POINT {
            tracing_write(
                memory,
                RV32_MEMORY_AS,
                record.rd_val + (i * POINT_SIZE) as u32,
                data[i],
                &mut record.writes_aux[i].prev_timestamp,
                &mut record.writes_aux[i].prev_data,
            );
        }
    }
}

impl<
        F: PrimeField32,
        const BLOCKS_PER_SCALAR: usize,
        const BLOCKS_PER_POINT: usize,
        const SCALAR_SIZE: usize,
        const POINT_SIZE: usize,
    > AdapterTraceFiller<F>
    for Rv32EcMulAdapterFiller<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>
{
    const WIDTH: usize = Rv32EcMulAdapterCols::<
        F,
        BLOCKS_PER_SCALAR,
        BLOCKS_PER_POINT,
        SCALAR_SIZE,
        POINT_SIZE,
    >::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv32EcMulAdapterRecord<
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv32EcMulAdapterCols<
            F,
            BLOCKS_PER_SCALAR,
            BLOCKS_PER_POINT,
            SCALAR_SIZE,
            POINT_SIZE,
        > = adapter_row.borrow_mut();

        // Range checks:
        // **NOTE**: Must do the range checks before overwriting the records
        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: usize = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

        self.bitwise_lookup_chip.request_range(
            (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
            (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits,
        );
        self.bitwise_lookup_chip.request_range(
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
        );

        let timestamp_delta = 3 + BLOCKS_PER_SCALAR + 2 * BLOCKS_PER_POINT;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // **NOTE**: Must iterate everything in reverse order to avoid overwriting the records
        record
            .writes_aux
            .iter()
            .rev()
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|(write, cols_write)| {
                cols_write.set_prev_data(write.prev_data.map(F::from_canonical_u8));
                mem_helper.fill(write.prev_timestamp, timestamp_mm(), cols_write.as_mut());
            });

        record
            .reads_point_aux
            .iter()
            .zip(cols.reads_point_aux.iter_mut())
            .rev()
            .for_each(|(read, cols_read)| {
                mem_helper.fill(read.prev_timestamp, timestamp_mm(), cols_read.as_mut());
            });

        record
            .reads_scalar_aux
            .iter()
            .zip(cols.reads_scalar_aux.iter_mut())
            .rev()
            .for_each(|(read, cols_read)| {
                mem_helper.fill(read.prev_timestamp, timestamp_mm(), cols_read.as_mut());
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

        cols.rd_val = record.rd_val.to_le_bytes().map(F::from_canonical_u8);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = val.to_le_bytes().map(F::from_canonical_u8);
            });
        cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_canonical_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
