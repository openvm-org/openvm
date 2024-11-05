use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
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
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers
///   are read from registers (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size
///   `BLOCK_SIZE` from the heap, starting from the addresses in `rs[0]`
///   (and `rs[1]` if `R = 2`).
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
    pub rs_read_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T, BLOCK_SIZE>; BLOCKS_PER_READ]; NUM_READS],

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
        let d = AB::F::from_canonical_usize(1);
        let e = AB::F::from_canonical_usize(2);

        // Read register values for rs
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), val, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with
        // a range check on the highest limb.
        let rs_val_f = cols.rs_val.map(|decomp| {
            // TODO: range check
            decomp.iter().rev().fold(AB::Expr::zero(), |acc, &limb| {
                acc * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS) + limb
            })
        });

        let need_range_check: [_; 2] = from_fn(|i| {
            if i < NUM_READS {
                cols.rs_val[i][RV32_REGISTER_NUM_LIMBS - 1].into()
            } else {
                AB::Expr::zero()
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
                        .unwrap_or(AB::Expr::zero()),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::zero()),
                    d.into(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[derive(Debug)]
pub struct Rv32IsEqualModAdapterChip<
    F: Field,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub air: Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>,
    pub bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
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
        memory_controller: MemoryControllerRef<F>,
        bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let address_bits = memory_controller.mem_config.pointer_max_bits;
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

#[derive(Clone, Debug)]
pub struct Rv32IsEqualModReadRecord<
    F: Field,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    pub rs: [MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>; NUM_READS],
    pub reads: [[MemoryReadRecord<F, BLOCK_SIZE>; BLOCKS_PER_READ]; NUM_READS],
}

#[derive(Clone, Debug)]
pub struct Rv32IsEqualModWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LIMBS>,
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
    type ReadRecord = Rv32IsEqualModReadRecord<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>;
    type WriteRecord = Rv32IsEqualModWriteRecord<F>;
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

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

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
                memory
                    .read::<BLOCK_SIZE>(e, F::from_canonical_u32(address + (i * BLOCK_SIZE) as u32))
            })
        });

        let read_data = read_records.map(|r| {
            let read = r.map(|x| x.data);
            let mut read_it = read.iter().flatten();
            from_fn(|_| *(read_it.next().unwrap()))
        });
        let record = Rv32IsEqualModReadRecord {
            rs: rs_records,
            reads: read_records,
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
        let rd = memory.write(d, a, output.writes[0]);

        debug_assert!(
            memory.timestamp() - from_state.timestamp
                == (NUM_READS * (BLOCKS_PER_READ + 1) + 1) as u32,
            "timestamp delta is {}, expected {}",
            memory.timestamp() - from_state.timestamp,
            NUM_READS * (BLOCKS_PER_READ + 1) + 1
        );

        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, rd },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.rs_ptr = read_record.rs.map(|r| r.pointer);
        row_slice.rs_val = read_record.rs.map(|r| r.data);
        row_slice.rs_read_aux = read_record
            .rs
            .map(|r| aux_cols_factory.make_read_aux_cols(r));
        row_slice.heap_read_aux =
            read_record
                .reads
                .map(|r: [MemoryReadRecord<F, BLOCK_SIZE>; BLOCKS_PER_READ]| {
                    r.map(|x| aux_cols_factory.make_read_aux_cols(x))
                });

        row_slice.rd_ptr = write_record.rd.pointer;
        row_slice.writes_aux = aux_cols_factory.make_write_aux_cols(write_record.rd);

        // Range checks
        let need_range_check: [u32; 2] = from_fn(|i| {
            if i < NUM_READS {
                read_record.rs[i].data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
            } else {
                0
            }
        });
        let limb_shift = (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.air.address_bits) as u32;
        self.bitwise_lookup_chip.request_range(
            need_range_check[0] * limb_shift,
            need_range_check[1] * limb_shift,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
