use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::once,
    marker::PhantomData,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, ImmInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
            MemoryAddress, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
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
    read_rv32_register, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// This adapter reads from NUM_READS <= 2 pointers.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads are from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32HeapBranchAdapterCols<T, const NUM_READS: usize, const READ_SIZE: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],

    pub heap_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
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

        let d = AB::F::from_canonical_u32(RV32_REGISTER_AS);
        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        for (ptr, data, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), data, timestamp_pp(), aux)
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
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32HeapBranchAdapterCols<_, NUM_READS, READ_SIZE> = local.borrow();
        cols.from_state.pc
    }
}

pub struct Rv32HeapBranchAdapterChip<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    pub air: Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize>
    Rv32HeapBranchAdapterChip<F, NUM_READS, READ_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits < RV32_CELL_BITS,
            "address_bits={address_bits} needs to be large enough for high limb range check"
        );
        Self {
            air: Rv32HeapBranchAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bus: bitwise_lookup_chip.bus(),
                address_bits,
            },
            bitwise_lookup_chip,
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32HeapBranchReadRecord<const NUM_READS: usize, const READ_SIZE: usize> {
    #[serde(with = "BigArray")]
    pub rs_reads: [RecordId; NUM_READS],
    #[serde(with = "BigArray")]
    pub heap_reads: [RecordId; NUM_READS],
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize> VmAdapterChip<F>
    for Rv32HeapBranchAdapterChip<F, NUM_READS, READ_SIZE>
{
    type ReadRecord = Rv32HeapBranchReadRecord<NUM_READS, READ_SIZE>;
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

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let mut rs_vals = [0; NUM_READS];
        let rs_records: [_; NUM_READS] = from_fn(|i| {
            let addr = if i == 0 { a } else { b };
            let (record, val) = read_rv32_register(memory, d, addr);
            rs_vals[i] = val;
            record
        });

        let heap_records = rs_vals.map(|address| {
            assert!(address as usize + READ_SIZE - 1 < (1 << self.air.address_bits));
            memory.read::<u8, READ_SIZE>(e, F::from_canonical_u32(address))
        });

        let record = Rv32HeapBranchReadRecord {
            rs_reads: rs_records,
            heap_reads: heap_records.map(|r| r.0),
        };
        Ok((heap_records.map(|r| r.1.map(F::from_canonical_u8)), record))
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
            "timestamp delta is {}, expected 4",
            timestamp_delta
        );

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
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
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let row_slice: &mut Rv32HeapBranchAdapterCols<_, NUM_READS, READ_SIZE> =
            row_slice.borrow_mut();
        row_slice.from_state = write_record.map(F::from_canonical_u32);

        let rs_reads = read_record.rs_reads.map(|r| memory.record_by_id(r));

        for (i, rs_read) in rs_reads.iter().enumerate() {
            row_slice.rs_ptr[i] = rs_read.pointer;
            row_slice.rs_val[i].copy_from_slice(rs_read.data_slice());
            aux_cols_factory.generate_read_aux(rs_read, &mut row_slice.rs_read_aux[i]);
        }

        for (i, heap_read) in read_record.heap_reads.iter().enumerate() {
            let record = memory.record_by_id(*heap_read);
            aux_cols_factory.generate_read_aux(record, &mut row_slice.heap_read_aux[i]);
        }

        // Range checks:
        let need_range_check: Vec<u32> = rs_reads
            .iter()
            .map(|record| {
                record
                    .data_at(RV32_REGISTER_NUM_LIMBS - 1)
                    .as_canonical_u32()
            })
            .chain(once(0)) // in case NUM_READS is odd
            .collect();
        debug_assert!(self.air.address_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.air.address_bits;
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] << limb_shift_bits, pair[1] << limb_shift_bits);
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
