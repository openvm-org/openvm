use std::{array::from_fn, borrow::Borrow};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::{MEMORY_AS, REGISTER_AS};
use openvm_riscv_circuit::adapters::{
    eval_byte_ptr_limbs_to_block_index, expand_to_block, reg_byte_ptr_to_cell_ptr_limbs,
    PTR_U16_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
};

/// U16-shaped equality-mod adapter. Heap reads and register writes are
/// `BLOCK_FE_WIDTH` u16 cells per memory-bus message.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct IsEqualModU16AdapterCols<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; PTR_U16_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(IsEqualModU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ>)]
pub struct IsEqualModU16AdapterAir<
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
    > BaseAir<F> for IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB> for IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        BLOCK_FE_WIDTH,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        const {
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * BLOCK_FE_WIDTH,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * BLOCK_FE_WIDTH"
            );
        }
        let cols: &IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_u32(REGISTER_AS);
        let e = AB::F::from_u32(MEMORY_AS);

        // Read register values for rs (register pointers are small).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(d, reg_byte_ptr_to_cell_ptr_limbs::<AB>(ptr)),
                    expand_to_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let byte_ptr_max_bits = self.pointer_max_bits;

        // Convert each base *byte* pointer to the bus address of its first heap block,
        // enforcing eight-byte alignment.
        let rs_base: [MemoryAddress<AB::F, AB::Expr>; NUM_READS] = from_fn(|i| {
            MemoryAddress::new(
                e,
                eval_byte_ptr_limbs_to_block_index::<AB>(
                    builder,
                    self.range_bus,
                    cols.rs_val[i].map(Into::into),
                    byte_ptr_max_bits,
                    ctx.instruction.is_valid.clone(),
                ),
            )
        });

        // Reads from heap: block `j` is `j` blocks after the base address.
        let read_block_data: [[[_; BLOCK_FE_WIDTH]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });

        for (base, block_data, block_aux) in izip!(rs_base, read_block_data, &cols.heap_read_aux) {
            for (j, (data, aux)) in izip!(block_data, block_aux).enumerate() {
                self.memory_bridge
                    .read(base.offset_blocks(j), data, timestamp_pp(), aux)
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register (register pointer is small).
        self.memory_bridge
            .write(
                MemoryAddress::new(d, reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.rd_ptr)),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc_idx(
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
                (1, ctx.to_pc_idx),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc_idx(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        cols.from_state.pc
    }
}
