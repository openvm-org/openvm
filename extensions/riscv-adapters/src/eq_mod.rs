use std::{array::from_fn, borrow::Borrow};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{pack_u8_block, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr, expand_to_rv64_block, ptr_bound_from_high_u16_expr, u16_limbs_to_ptr,
    RV64_PTR_BITS, RV64_PTR_U16_LIMBS, RV64_REGISTER_NUM_LIMBS, U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive heap reads.
/// * Writes are to 64-bit register rd (8 bytes).
///
/// The materialized pointer values are stored as two u16 cells.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64IsEqualModAdapterCols<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64IsEqualModAdapterCols<u8, NUM_READS, BLOCKS_PER_READ>)]
pub struct Rv64IsEqualModAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F> for Rv64IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv64IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB> for Rv64IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        RV64_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        const {
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * MEMORY_BLOCK_BYTES,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * MEMORY_BLOCK_BYTES"
            );
        }
        let cols: &Rv64IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_u32(RV64_REGISTER_AS);
        let e = AB::F::from_u32(RV64_MEMORY_AS);

        // Read register values for rs.
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(d, byte_ptr_to_u16_ptr::<AB>(ptr)),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the two u16 cells into low 32-bit heap pointers.
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(|limbs| u16_limbs_to_ptr(&limbs));

        // Each materialized pointer is stored as two u16 cells. Bound the high
        // cell against the guest byte-pointer limit.
        for val in cols.rs_val.iter() {
            self.range_bus
                .range_check(
                    ptr_bound_from_high_u16_expr(val[1], self.pointer_max_bits),
                    U16_BITS,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Reads from heap
        let read_block_data: [[[_; MEMORY_BLOCK_BYTES]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });
        let block_ptr_offset: [_; BLOCKS_PER_READ] =
            from_fn(|i| AB::F::from_usize(i * MEMORY_BLOCK_BYTES));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, byte_ptr_to_u16_ptr::<AB>(ptr.clone() + offset)),
                        pack_u8_block::<AB>(&data),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register
        self.memory_bridge
            .write(
                MemoryAddress::new(d, byte_ptr_to_u16_ptr::<AB>(cols.rd_ptr)),
                pack_u8_block::<AB>(&ctx.writes[0].clone()),
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
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, Copy)]
pub struct Rv64IsEqualModAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64IsEqualModAdapterFiller<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const TOTAL_READ_SIZE: usize>
    Rv64IsEqualModAdapterExecutor<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        const {
            assert!(NUM_READS <= 2);
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * MEMORY_BLOCK_BYTES,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * MEMORY_BLOCK_BYTES"
            );
        }
        assert!(
            (U16_BITS..=RV64_PTR_BITS).contains(&pointer_max_bits),
            "pointer_max_bits must be in [16, 32]"
        );
        Self { pointer_max_bits }
    }
}
