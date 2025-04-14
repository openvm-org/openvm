use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
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
    read_rv32_register, tmp_convert_to_u8s, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use serde_with::serde_as;

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size `BLOCK_SIZE` from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes are to 32-bit register rd.
#[repr(C)]
#[derive(AlignedBorrow)]
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

pub struct Rv32IsEqualModAdapterChip<
    F: Field,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub air: Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > Rv32IsEqualModAdapterChip<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits < RV32_CELL_BITS,
            "address_bits={address_bits} needs to be large enough for high limb range check"
        );
        Self {
            air: Rv32IsEqualModAdapterAir {
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rv32IsEqualModReadRecord<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    #[serde(with = "BigArray")]
    pub rs: [RecordId; NUM_READS],
    #[serde_as(as = "[[_; BLOCKS_PER_READ]; NUM_READS]")]
    pub reads: [[RecordId; BLOCKS_PER_READ]; NUM_READS],
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rv32IsEqualModWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub rd_id: RecordId,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterChip<F>
    for Rv32IsEqualModAdapterChip<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    type ReadRecord = Rv32IsEqualModReadRecord<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>;
    type WriteRecord = Rv32IsEqualModWriteRecord;
    type Air = Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let mut rs_vals = [0; NUM_READS];
        let rs_records: [_; NUM_READS] = from_fn(|i| {
            let addr = if i == 0 { b } else { c };
            let (record, val) = read_rv32_register(memory, d, addr);
            rs_vals[i] = val;
            record
        });

        let read_records = rs_vals.map(|address| {
            debug_assert!(address < (1 << self.air.address_bits));
            from_fn(|i| {
                memory.read::<u8, BLOCK_SIZE>(
                    e,
                    F::from_canonical_u32(address + (i * BLOCK_SIZE) as u32),
                )
            })
        });

        let read_data = read_records.map(|r| {
            let read = r.map(|x| x.1);
            let mut read_it = read.iter().flatten();
            from_fn(|_| *(read_it.next().unwrap())).map(F::from_canonical_u8)
        });
        let record = Rv32IsEqualModReadRecord {
            rs: rs_records,
            reads: read_records.map(|r| r.map(|x| x.0)),
        };

        Ok((read_data, record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { a, d, .. } = *instruction;
        let (rd_id, _) = memory.write(d, a, &tmp_convert_to_u8s(output.writes[0]));

        debug_assert!(
            memory.timestamp() - from_state.timestamp
                == (NUM_READS * (BLOCKS_PER_READ + 1) + 1) as u32,
            "timestamp delta is {}, expected {}",
            memory.timestamp() - from_state.timestamp,
            NUM_READS * (BLOCKS_PER_READ + 1) + 1
        );

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, rd_id },
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
        let row_slice: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        let rs = read_record.rs.map(|r| memory.record_by_id(r));
        for (i, r) in rs.iter().enumerate() {
            row_slice.rs_ptr[i] = r.pointer;
            row_slice.rs_val[i].copy_from_slice(r.data_slice());
            aux_cols_factory.generate_read_aux(r, &mut row_slice.rs_read_aux[i]);
            for (j, x) in read_record.reads[i].iter().enumerate() {
                let read = memory.record_by_id(*x);
                aux_cols_factory.generate_read_aux(read, &mut row_slice.heap_read_aux[i][j]);
            }
        }

        let rd = memory.record_by_id(write_record.rd_id);
        row_slice.rd_ptr = rd.pointer;
        aux_cols_factory.generate_write_aux(rd, &mut row_slice.writes_aux);

        // Range checks
        let need_range_check: [u32; 2] = from_fn(|i| {
            if i < NUM_READS {
                rs[i]
                    .data_at(RV32_REGISTER_NUM_LIMBS - 1)
                    .as_canonical_u32()
            } else {
                0
            }
        });
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.air.address_bits;
        self.bitwise_lookup_chip.request_range(
            need_range_check[0] << limb_shift_bits,
            need_range_check[1] << limb_shift_bits,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
