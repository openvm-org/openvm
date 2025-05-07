use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::zip,
    marker::PhantomData,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VecHeapTwoReadsAdapterInterface, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
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
    abstract_compose, read_rv32_register, tmp_convert_to_u8s, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// This adapter reads from 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READX` consecutive reads of size `READ_SIZE` from the heap,
///   starting from the addresses in `rs[X]`
/// * NOTE that the two reads can read different numbers of blocks.
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive writes of size `WRITE_SIZE` to the
///   heap, starting from the address in `rd`.
pub struct Rv32VecHeapTwoReadsAdapterChip<
    F: Field,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<
        F: PrimeField32,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    >
    Rv32VecHeapTwoReadsAdapterChip<
        F,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits < RV32_CELL_BITS,
            "address_bits={address_bits} needs to be large enough for high limb range check"
        );
        Self {
            air: Rv32VecHeapTwoReadsAdapterAir {
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
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Rv32VecHeapTwoReadsReadRecord<
    F: Field,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const READ_SIZE: usize,
> {
    /// Read register value from address space e=1
    pub rs1: RecordId,
    pub rs2: RecordId,
    /// Read register value from address space d=1
    pub rd: RecordId,

    pub rd_val: F,

    #[serde_as(as = "[_; BLOCKS_PER_READ1]")]
    pub reads1: [RecordId; BLOCKS_PER_READ1],
    #[serde_as(as = "[_; BLOCKS_PER_READ2]")]
    pub reads2: [RecordId; BLOCKS_PER_READ2],
}

#[repr(C)]
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32VecHeapTwoReadsWriteRecord<const BLOCKS_PER_WRITE: usize, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<u32>,
    #[serde_as(as = "[_; BLOCKS_PER_WRITE]")]
    pub writes: [RecordId; BLOCKS_PER_WRITE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32VecHeapTwoReadsAdapterCols<
    T,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub rd_ptr: T,

    pub rs1_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs2_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs1_read_aux: MemoryReadAuxCols<T>,
    pub rs2_read_aux: MemoryReadAuxCols<T>,
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads1_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_READ1],
    pub reads2_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_READ2],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32VecHeapTwoReadsAdapterAir<
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F>
    for Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    fn width(&self) -> usize {
        Rv32VecHeapTwoReadsAdapterCols::<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >::width()
    }
}

impl<
        AB: InteractionBuilder,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Interface = VecHeapTwoReadsAdapterInterface<
        AB::Expr,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32VecHeapTwoReadsAdapterCols<
            _,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let ptrs = [cols.rs1_ptr, cols.rs2_ptr, cols.rd_ptr];
        let vals = [cols.rs1_val, cols.rs2_val, cols.rd_val];
        let auxs = [&cols.rs1_read_aux, &cols.rs2_read_aux, &cols.rd_read_aux];

        // Read register values for rs1, rs2, rd
        for (ptr, val, aux) in izip!(ptrs, vals, auxs) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), ptr),
                    val,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Range checks: see Rv32VecHeaperAdapterAir
        let need_range_check = [&cols.rs1_val, &cols.rs2_val, &cols.rd_val, &cols.rd_val]
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1]);

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

        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs1_val_f: AB::Expr = abstract_compose(cols.rs1_val);
        let rs2_val_f: AB::Expr = abstract_compose(cols.rs2_val);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);
        // Reads from heap
        for (i, (read, aux)) in zip(ctx.reads.0, &cols.reads1_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        rs1_val_f.clone() + AB::Expr::from_canonical_usize(i * READ_SIZE),
                    ),
                    read,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        for (i, (read, aux)) in zip(ctx.reads.1, &cols.reads2_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        rs2_val_f.clone() + AB::Expr::from_canonical_usize(i * READ_SIZE),
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
                    cols.rs1_ptr.into(),
                    cols.rs2_ptr.into(),
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
        let cols: &Rv32VecHeapTwoReadsAdapterCols<
            _,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        cols.from_state.pc
    }
}

impl<
        F: PrimeField32,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterChip<F>
    for Rv32VecHeapTwoReadsAdapterChip<
        F,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type ReadRecord =
        Rv32VecHeapTwoReadsReadRecord<F, BLOCKS_PER_READ1, BLOCKS_PER_READ2, READ_SIZE>;
    type WriteRecord = Rv32VecHeapTwoReadsWriteRecord<BLOCKS_PER_WRITE, WRITE_SIZE>;
    type Air = Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >;
    type Interface = VecHeapTwoReadsAdapterInterface<
        F,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let (rs1_record, rs1_val) = read_rv32_register(memory, d, b);
        let (rs2_record, rs2_val) = read_rv32_register(memory, d, c);
        let (rd_record, rd_val) = read_rv32_register(memory, d, a);

        assert!(rs1_val as usize + READ_SIZE * BLOCKS_PER_READ1 - 1 < (1 << self.air.address_bits));
        let read1_records = from_fn(|i| {
            memory.read::<u8, READ_SIZE>(e, F::from_canonical_u32(rs1_val + (i * READ_SIZE) as u32))
        });
        let read1_data = read1_records.map(|r| r.1.map(F::from_canonical_u8));
        assert!(rs2_val as usize + READ_SIZE * BLOCKS_PER_READ2 - 1 < (1 << self.air.address_bits));
        let read2_records = from_fn(|i| {
            memory.read::<u8, READ_SIZE>(e, F::from_canonical_u32(rs2_val + (i * READ_SIZE) as u32))
        });
        let read2_data = read2_records.map(|r| r.1.map(F::from_canonical_u8));
        assert!(rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1 < (1 << self.air.address_bits));

        let record = Rv32VecHeapTwoReadsReadRecord {
            rs1: rs1_record,
            rs2: rs2_record,
            rd: rd_record,
            rd_val: F::from_canonical_u32(rd_val),
            reads1: read1_records.map(|r| r.0),
            reads2: read2_records.map(|r| r.0),
        };

        Ok(((read1_data, read2_data), record))
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
            let (record_id, _) = memory.write(
                e,
                read_record.rd_val + F::from_canonical_u32((i * WRITE_SIZE) as u32),
                &tmp_convert_to_u8s(write),
            );
            i += 1;
            record_id
        });

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
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
        memory: &OfflineMemory<F>,
    ) {
        vec_heap_two_reads_generate_trace_row_impl(
            row_slice,
            &read_record,
            &write_record,
            self.bitwise_lookup_chip.clone(),
            self.air.address_bits,
            memory,
        )
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn vec_heap_two_reads_generate_trace_row_impl<
    F: PrimeField32,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(
    row_slice: &mut [F],
    read_record: &Rv32VecHeapTwoReadsReadRecord<F, BLOCKS_PER_READ1, BLOCKS_PER_READ2, READ_SIZE>,
    write_record: &Rv32VecHeapTwoReadsWriteRecord<BLOCKS_PER_WRITE, WRITE_SIZE>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    address_bits: usize,
    memory: &OfflineMemory<F>,
) {
    let aux_cols_factory = memory.aux_cols_factory();
    let row_slice: &mut Rv32VecHeapTwoReadsAdapterCols<
        F,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    > = row_slice.borrow_mut();
    row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

    let rd = memory.record_by_id(read_record.rd);
    let rs1 = memory.record_by_id(read_record.rs1);
    let rs2 = memory.record_by_id(read_record.rs2);

    row_slice.rd_ptr = rd.pointer;
    row_slice.rs1_ptr = rs1.pointer;
    row_slice.rs2_ptr = rs2.pointer;

    row_slice.rd_val.copy_from_slice(rd.data_slice());
    row_slice.rs1_val.copy_from_slice(rs1.data_slice());
    row_slice.rs2_val.copy_from_slice(rs2.data_slice());

    aux_cols_factory.generate_read_aux(rs1, &mut row_slice.rs1_read_aux);
    aux_cols_factory.generate_read_aux(rs2, &mut row_slice.rs2_read_aux);
    aux_cols_factory.generate_read_aux(rd, &mut row_slice.rd_read_aux);

    for (i, r) in read_record.reads1.iter().enumerate() {
        let record = memory.record_by_id(*r);
        aux_cols_factory.generate_read_aux(record, &mut row_slice.reads1_aux[i]);
    }

    for (i, r) in read_record.reads2.iter().enumerate() {
        let record = memory.record_by_id(*r);
        aux_cols_factory.generate_read_aux(record, &mut row_slice.reads2_aux[i]);
    }

    for (i, w) in write_record.writes.iter().enumerate() {
        let record = memory.record_by_id(*w);
        aux_cols_factory.generate_write_aux(record, &mut row_slice.writes_aux[i]);
    }
    // Range checks:
    let need_range_check = [
        &read_record.rs1,
        &read_record.rs2,
        &read_record.rd,
        &read_record.rd,
    ]
    .map(|record| {
        memory
            .record_by_id(*record)
            .data_at(RV32_REGISTER_NUM_LIMBS - 1)
            .as_canonical_u32()
    });
    debug_assert!(address_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
    let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits;
    for pair in need_range_check.chunks_exact(2) {
        bitwise_lookup_chip.request_range(pair[0] << limb_shift_bits, pair[1] << limb_shift_bits);
    }
}
