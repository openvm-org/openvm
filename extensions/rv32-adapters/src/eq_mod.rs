use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::adapters::{
    memory_read, memory_write, new_read_rv32_register, tracing_read, tracing_write, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size `BLOCK_SIZE` from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes are to 32-bit register rd.
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32IsEqualModAdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32IsEqualModAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F>
    for Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv32IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_canonical_u32(RV32_REGISTER_AS);
        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        // Read register values for rs
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), val, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with
        // a range check on the highest limb.
        let rs_val_f = cols.rs_val.map(|decomp| {
            decomp.iter().rev().fold(AB::Expr::ZERO, |acc, &limb| {
                acc * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS) + limb
            })
        });

        let need_range_check: [_; 2] = from_fn(|i| {
            if i < NUM_READS {
                cols.rs_val[i][RV32_REGISTER_NUM_LIMBS - 1].into()
            } else {
                AB::Expr::ZERO
            }
        });

        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        self.bus
            .send_range(
                need_range_check[0].clone() * limb_shift,
                need_range_check[1].clone() * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Reads from heap
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        let read_block_data: [[[_; BLOCK_SIZE]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });
        let block_ptr_offset: [_; BLOCKS_PER_READ] =
            from_fn(|i| AB::F::from_canonical_usize(i * BLOCK_SIZE));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, ptr.clone() + offset),
                        data,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register
        self.memory_bridge
            .write(
                MemoryAddress::new(d, cols.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

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
                    d.into(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

pub struct Rv32IsEqualModeAdapterStep<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > Rv32IsEqualModeAdapterStep<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
        );
        Self {
            pointer_max_bits,
            bitwise_lookup_chip,
        }
    }
}

impl<
        F: PrimeField32,
        CTX,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceStep<F, CTX>
    for Rv32IsEqualModeAdapterStep<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize =
        Rv32IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let cols: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();
        cols.from_state.pc = F::from_canonical_u32(pc);
        cols.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction { b, c, d, e, .. } = *instruction;

        let e = e.as_canonical_u32();
        let d = d.as_canonical_u32();
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        let cols: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();

        // Read register values
        let rs_vals: [_; NUM_READS] = from_fn(|i| {
            let addr = if i == 0 { b } else { c };
            cols.rs_ptr[i] = addr;
            let rs_val = tracing_read(memory, d, addr.as_canonical_u32(), &mut cols.rs_read_aux[i]);
            cols.rs_val[i] = rs_val.map(F::from_canonical_u8);
            u32::from_le_bytes(rs_val)
        });

        // Read memory values
        from_fn(|i| {
            assert!(rs_vals[i] as usize + TOTAL_READ_SIZE - 1 < (1 << self.pointer_max_bits));
            from_fn::<_, BLOCKS_PER_READ, _>(|j| {
                tracing_read::<_, BLOCK_SIZE>(
                    memory,
                    e,
                    rs_vals[i] + (j * BLOCK_SIZE) as u32,
                    &mut cols.heap_read_aux[i][j],
                )
            })
            .concat()
            .try_into()
            .unwrap()
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction { a, d, .. } = *instruction;
        let cols: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();
        cols.rd_ptr = a;
        tracing_write(
            memory,
            d.as_canonical_u32(),
            a.as_canonical_u32(),
            data,
            &mut cols.writes_aux,
        );
    }

    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _ctx: (),
        adapter_row: &mut [F],
    ) {
        let cols: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();
        let mut timestamp = cols.from_state.timestamp.as_canonical_u32();
        let mut timestamp_pp = || {
            timestamp += 1;
            timestamp - 1
        };

        cols.rs_read_aux.iter_mut().for_each(|aux| {
            mem_helper.fill_from_prev(timestamp_pp(), aux.as_mut());
        });

        cols.heap_read_aux.iter_mut().for_each(|reads| {
            reads
                .iter_mut()
                .for_each(|aux| mem_helper.fill_from_prev(timestamp_pp(), aux.as_mut()));
        });

        mem_helper.fill_from_prev(timestamp_pp(), cols.writes_aux.as_mut());

        // Range checks:
        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        self.bitwise_lookup_chip.request_range(
            cols.rs_val[0][RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits,
            if NUM_READS > 1 {
                cols.rs_val[1][RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits
            } else {
                0
            },
        );
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterExecutorE1<F>
    for Rv32IsEqualModeAdapterStep<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    fn read(&self, memory: &mut GuestMemory, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { b, c, d, e, .. } = *instruction;

        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        // Read register values
        let rs_vals = from_fn(|i| {
            let addr = if i == 0 { b } else { c };
            new_read_rv32_register(memory, d, addr.as_canonical_u32())
        });

        // Read memory values
        rs_vals.map(|address| {
            assert!(address as usize + TOTAL_READ_SIZE - 1 < (1 << self.pointer_max_bits));
            memory_read(memory, e, address)
        })
    }

    fn write(
        &self,
        memory: &mut GuestMemory,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) {
        let Instruction { a, d, .. } = *instruction;
        memory_write(memory, d.as_canonical_u32(), a.as_canonical_u32(), data);
    }
}
