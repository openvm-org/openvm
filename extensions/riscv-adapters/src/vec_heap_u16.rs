use std::{array::from_fn, borrow::Borrow, iter::once};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir,
        BLOCK_FE_WIDTH,
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
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    eval_byte_ptr_limbs_to_block_index, expand_to_block, reg_byte_ptr_to_cell_ptr_limbs,
    PTR_U16_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
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
pub struct VecHeapU16AdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    /// Low 32 bits of rs registers as little-endian 16-bit *byte*-pointer limbs.
    pub rs_val: [[T; PTR_U16_LIMBS]; NUM_READS],
    /// Low 32 bits of rd register as little-endian 16-bit *byte*-pointer limbs.
    pub rd_val: [T; PTR_U16_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(VecHeapU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>)]
pub struct VecHeapU16AdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    /// Maximum bit width of guest byte pointers.
    pointer_max_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > BaseAir<F> for VecHeapU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    fn width(&self) -> usize {
        VecHeapU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > VmAdapterAir<AB> for VecHeapU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
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
        let cols: &VecHeapU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd (register pointers are small).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            let bus_payload: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_block(&val);
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(REGISTER_AS),
                        reg_byte_ptr_to_cell_ptr_limbs::<AB>(ptr),
                    ),
                    bus_payload,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let byte_ptr_max_bits = self.pointer_max_bits;
        let e = AB::F::from_u32(MEMORY_AS);

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
        let rd_base = MemoryAddress::new(
            e,
            eval_byte_ptr_limbs_to_block_index::<AB>(
                builder,
                self.range_bus,
                cols.rd_val.map(Into::into),
                byte_ptr_max_bits,
                ctx.instruction.is_valid.clone(),
            ),
        );

        // Reads from heap: block `j` is `j` blocks after the base address.
        for (base, reads, reads_aux) in izip!(rs_base, ctx.reads, &cols.reads_aux) {
            for (j, (read, aux)) in izip!(reads, reads_aux).enumerate() {
                let read_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|k| read[k].clone());
                self.memory_bridge
                    .read(base.offset_blocks(j), read_array, timestamp_pp(), aux)
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap: block `j` is `j` blocks after the base address.
        for (j, (write, aux)) in izip!(ctx.writes, &cols.writes_aux).enumerate() {
            let write_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|k| write[k].clone());
            self.memory_bridge
                .write(rd_base.offset_blocks(j), write_array, timestamp_pp(), aux)
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
                    AB::Expr::from_u32(REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &VecHeapU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        cols.from_state.pc
    }
}
