use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    iter::once,
    marker::PhantomData,
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::instruction::Instruction;
use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{read_rv32_register, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, ImmInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord,
        },
        program::ProgramBus,
    },
};

/// This adapter reads from NUM_READS <= 2 pointers.
/// * The data is read from the heap (address space 2), and the pointers
///   are read from registers (address space 1).
/// * Reads are from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32HeapBranchAdapterCols<T, const NUM_READS: usize, const READ_SIZE: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>; NUM_READS],

    pub heap_read_aux: [MemoryReadAuxCols<T, READ_SIZE>; NUM_READS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HeapBranchAdapterAir<const NUM_READS: usize, const READ_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    address_bits: usize,
}

impl<F: Field, const NUM_READS: usize, const READ_SIZE: usize> BaseAir<F>
    for Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>
{
    fn width(&self) -> usize {
        Rv32HeapBranchAdapterCols::<F, NUM_READS, READ_SIZE>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_READS: usize, const READ_SIZE: usize> VmAdapterAir<AB>
    for Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, NUM_READS, 0, READ_SIZE, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32HeapBranchAdapterCols<_, NUM_READS, READ_SIZE> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let d = AB::F::ONE;
        let e = AB::F::from_canonical_usize(2);

        for (ptr, data, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), data, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // We constrain the highest limbs of heap pointers to be less than 2^(addr_bits - (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))).
        // This ensures that no overflow occurs when computing memory pointers. Since the number of cells accessed with each address
        // will be small enough, and combined with the memory argument, it ensures that all the cells accessed in the memory are less than 2^addr_bits.
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1])
            .collect();

        // range checks constrain to RV32_CELL_BITS bits, so we need to shift the limbs to constrain the correct amount of bits
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        // Note: since limbs are read from memory we alread know that limb[i] < 2^RV32_CELL_BITS
        //       thus range checking limb[i] * shift < 2^RV32_CELL_BITS, gives us that
        //       limb[i] < 2^(addr_bits - (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
        for pair in need_range_check.chunks(2) {
            self.bus
                .send_range(
                    pair[0] * limb_shift,
                    pair.get(1).map(|x| (*x).into()).unwrap_or(AB::Expr::ZERO) * limb_shift, // in case NUM_READS is odd
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let heap_ptr = cols.rs_val.map(|r| {
            r.iter().rev().fold(AB::Expr::ZERO, |acc, limb| {
                acc * AB::F::from_canonical_u32(1 << RV32_CELL_BITS) + (*limb)
            })
        });
        for (ptr, data, aux) in izip!(heap_ptr, ctx.reads, &cols.heap_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(e, ptr), data, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
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
                    d.into(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32HeapBranchAdapterCols<_, NUM_READS, READ_SIZE> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Debug)]
pub struct Rv32HeapBranchAdapterChip<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    pub air: Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>,
    pub bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize>
    Rv32HeapBranchAdapterChip<F, NUM_READS, READ_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        let memory_controller = RefCell::borrow(&memory_controller);
        Self {
            air: Rv32HeapBranchAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge: memory_controller.memory_bridge(),
                bus: bitwise_lookup_chip.bus(),
                address_bits: memory_controller.mem_config.pointer_max_bits,
            },
            bitwise_lookup_chip,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32HeapBranchReadRecord<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    pub rs_reads: [MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>; NUM_READS],
    pub heap_reads: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize> VmAdapterChip<F>
    for Rv32HeapBranchAdapterChip<F, NUM_READS, READ_SIZE>
{
    type ReadRecord = Rv32HeapBranchReadRecord<F, NUM_READS, READ_SIZE>;
    type WriteRecord = ExecutionState<u32>;
    type Air = Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>;
    type Interface = BasicAdapterInterface<F, ImmInstruction<F>, NUM_READS, 0, READ_SIZE, 0>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

        let mut rs_vals = [0; NUM_READS];
        let rs_records: [_; NUM_READS] = from_fn(|i| {
            let addr = if i == 0 { a } else { b };
            let (record, val) = read_rv32_register(memory, d, addr);
            rs_vals[i] = val;
            record
        });

        let heap_records = rs_vals.map(|address| {
            assert!(address as usize + READ_SIZE - 1 < (1 << self.air.address_bits));
            memory.read::<READ_SIZE>(e, F::from_canonical_u32(address))
        });

        let record = Rv32HeapBranchReadRecord {
            rs_reads: rs_records,
            heap_reads: heap_records,
        };
        Ok((heap_records.map(|r| r.data), record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let timestamp_delta = memory.timestamp() - from_state.timestamp;
        debug_assert!(
            timestamp_delta == 4,
            "timestamp delta is {}, expected 2",
            timestamp_delta
        );

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            from_state,
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut Rv32HeapBranchAdapterCols<_, NUM_READS, READ_SIZE> =
            row_slice.borrow_mut();
        row_slice.from_state = write_record.map(F::from_canonical_u32);
        row_slice.rs_ptr = read_record.rs_reads.map(|r| r.pointer);
        row_slice.rs_val = read_record.rs_reads.map(|r| r.data);
        row_slice.rs_read_aux = read_record
            .rs_reads
            .map(|r| aux_cols_factory.make_read_aux_cols(r));
        row_slice.heap_read_aux = read_record
            .heap_reads
            .map(|r| aux_cols_factory.make_read_aux_cols(r));

        // Range checks:
        let need_range_check: Vec<u32> = read_record
            .rs_reads
            .iter()
            .map(|record| record.data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32())
            .chain(once(0)) // in case NUM_READS is odd
            .collect();
        debug_assert!(self.air.address_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift = (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.air.address_bits) as u32;
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] * limb_shift, pair[1] * limb_shift);
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
