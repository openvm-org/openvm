use std::{borrow::Borrow, iter::zip};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, VecHeapBranchAdapterInterface,
        VmAdapterAir, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{pack_u8_block, MemoryBridge, MemoryReadAuxCols},
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
    RV64_PTR_U16_LIMBS, U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
};

/// This adapter reads from NUM_READS <= 2 pointers (for branch operations).
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive `MEMORY_BLOCK_BYTES` reads from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `NUM_READS = 2`).
/// * No writes are performed (branch operations only compare values).
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64VecHeapBranchAdapterCols<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64VecHeapBranchAdapterCols<u8, NUM_READS, BLOCKS_PER_READ>)]
pub struct Rv64VecHeapBranchAdapterAir<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    /// Maximum bit width of guest byte pointers.
    pointer_max_bits: usize,
}

impl<F: Field, const NUM_READS: usize, const BLOCKS_PER_READ: usize> BaseAir<F>
    for Rv64VecHeapBranchAdapterAir<NUM_READS, BLOCKS_PER_READ>
{
    fn width(&self) -> usize {
        Rv64VecHeapBranchAdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_READS: usize, const BLOCKS_PER_READ: usize> VmAdapterAir<AB>
    for Rv64VecHeapBranchAdapterAir<NUM_READS, BLOCKS_PER_READ>
{
    type Interface =
        VecHeapBranchAdapterInterface<AB::Expr, NUM_READS, BLOCKS_PER_READ, MEMORY_BLOCK_BYTES>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv64VecHeapBranchAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read register values for rs
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(RV64_REGISTER_AS),
                        byte_ptr_to_u16_ptr::<AB>(ptr),
                    ),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

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

        // Compose the two u16 cells into low 32-bit heap pointers.
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(|limbs| u16_limbs_to_ptr(&limbs));

        let e = AB::F::from_u32(RV64_MEMORY_AS);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
                            byte_ptr_to_u16_ptr::<AB>(
                                address.clone() + AB::Expr::from_usize(i * MEMORY_BLOCK_BYTES),
                            ),
                        ),
                        pack_u8_block::<AB>(&read),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rs_ptr
                        .first()
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    ctx.instruction.immediate,
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
        let cols: &Rv64VecHeapBranchAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv64VecHeapBranchAdapterExecutor<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64VecHeapBranchAdapterFiller<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}
