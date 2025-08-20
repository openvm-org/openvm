use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{
                MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord,
                MemoryWriteAuxCols,
            },
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{memory_read_native, timed_write_native},
    },
};
use openvm_circuit_primitives::{
    utils::{not, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode, NATIVE_AS,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{memory_read, timed_write, tracing_read, RV32_CELL_BITS};

/// LoadStore Adapter handles all memory and register operations, so it must be aware
/// of the instruction type, specifically whether it is a load or store
/// LoadStore Adapter handles 4 byte aligned lw, sw instructions,
///                           2 byte aligned lh, lhu, sh instructions and
///                           1 byte aligned lb, lbu, sb instructions
/// This adapter always batch reads/writes 4 bytes,
/// thus it needs to shift left the memory pointer by some amount in case of not 4 byte aligned
/// intermediate pointers
pub struct LoadStoreInstruction<T> {
    /// is_valid is constrained to be bool
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    /// is_load is constrained to be bool, and can only be 1 if is_valid is 1
    pub is_load: T,

    /// Keeping two separate shift amounts is needed for getting the read_ptr/write_ptr with degree
    /// 2 load_shift_amount will be the shift amount if load and 0 if store
    pub load_shift_amount: T,
    /// store_shift_amount will be 0 if load and the shift amount if store
    pub store_shift_amount: T,
}

pub struct Rv32LoadStoreAdapterAirInterface<AB: InteractionBuilder>(PhantomData<AB>);

/// Using AB::Var for prev_data and AB::Expr for read_data
impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr> for Rv32LoadStoreAdapterAirInterface<AB> {
    type Reads = (
        [AB::Var; RV32_REGISTER_NUM_LIMBS],
        [AB::Expr; RV32_REGISTER_NUM_LIMBS],
    );
    type Writes = [[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1];
    type ProcessedInstruction = LoadStoreInstruction<AB::Expr>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32LoadStoreAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,

    /// Will write to rd when Load and read from rs2 when Store
    pub rd_rs2_ptr: T,
    pub read_data_aux: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    pub mem_ptr_limbs: [T; 2],
    pub mem_as: T,
    /// prev_data will be provided by the core chip to make a complete MemoryWriteAuxCols
    pub write_base_aux: MemoryBaseAuxCols<T>,
    /// Only writes if `needs_write`.
    /// If the instruction is a Load:
    /// - Sets `needs_write` to 0 iff `rd == x0`
    ///
    /// Otherwise:
    /// - Sets `needs_write` to 1
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32LoadStoreAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv32LoadStoreAdapterAir {
    fn width(&self) -> usize {
        Rv32LoadStoreAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32LoadStoreAdapterAir {
    type Interface = Rv32LoadStoreAdapterAirInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32LoadStoreAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let is_load = ctx.instruction.is_load;
        let is_valid = ctx.instruction.is_valid;
        let load_shift_amount = ctx.instruction.load_shift_amount;
        let store_shift_amount = ctx.instruction.store_shift_amount;
        let shift_amount = load_shift_amount.clone() + store_shift_amount.clone();

        let write_count = local_cols.needs_write;

        // This constraint ensures that the memory write only occurs when `is_valid == 1`.
        builder.assert_bool(write_count);
        builder.when(write_count).assert_one(is_valid.clone());

        // Constrain that if `is_valid == 1` and `write_count == 0`, then `is_load == 1` and
        // `rd_rs2_ptr == x0`
        builder
            .when(is_valid.clone() - write_count)
            .assert_one(is_load.clone());
        builder
            .when(is_valid.clone() - write_count)
            .assert_zero(local_cols.rd_rs2_ptr);

        // read rs1
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                local_cols.rs1_data,
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, is_valid.clone());

        // constrain mem_ptr = rs1 + imm as a u32 addition with 2 limbs
        let limbs_01 = local_cols.rs1_data[0]
            + local_cols.rs1_data[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let limbs_23 = local_cols.rs1_data[2]
            + local_cols.rs1_data[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);

        let inv = AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2)).inverse();
        let carry = (limbs_01 + local_cols.imm - local_cols.mem_ptr_limbs[0]) * inv;

        builder.when(is_valid.clone()).assert_bool(carry.clone());

        builder
            .when(is_valid.clone())
            .assert_bool(local_cols.imm_sign);
        let imm_extend_limb =
            local_cols.imm_sign * AB::F::from_canonical_u32((1 << (RV32_CELL_BITS * 2)) - 1);
        let carry = (limbs_23 + imm_extend_limb + carry - local_cols.mem_ptr_limbs[1]) * inv;
        builder.when(is_valid.clone()).assert_bool(carry.clone());

        // preventing mem_ptr overflow
        self.range_bus
            .range_check(
                // (limb[0] - shift_amount) / 4 < 2^14 => limb[0] - shift_amount < 2^16
                (local_cols.mem_ptr_limbs[0] - shift_amount)
                    * AB::F::from_canonical_u32(4).inverse(),
                RV32_CELL_BITS * 2 - 2,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(
                local_cols.mem_ptr_limbs[1],
                self.pointer_max_bits - RV32_CELL_BITS * 2,
            )
            .eval(builder, is_valid.clone());

        let mem_ptr = local_cols.mem_ptr_limbs[0]
            + local_cols.mem_ptr_limbs[1] * AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2));

        let is_store = is_valid.clone() - is_load.clone();
        // constrain mem_as to be in {0, 1, 2} if the instruction is a load,
        // and in {2, 3, 4} if the instruction is a store
        builder.assert_tern(local_cols.mem_as - is_store * AB::Expr::TWO);
        builder
            .when(not::<AB::Expr>(is_valid.clone()))
            .assert_zero(local_cols.mem_as);

        // read_as is [local_cols.mem_as] for loads and 1 for stores
        let read_as = select::<AB::Expr>(
            is_load.clone(),
            local_cols.mem_as,
            AB::F::from_canonical_u32(RV32_REGISTER_AS),
        );

        // read_ptr is mem_ptr for loads and rd_rs2_ptr for stores
        // Note: shift_amount is expected to have degree 2, thus we can't put it in the select
        // clause       since the resulting read_ptr/write_ptr's degree will be 3 which is
        // too high.       Instead, the solution without using additional columns is to get
        // two different shift amounts from core chip
        let read_ptr = select::<AB::Expr>(is_load.clone(), mem_ptr.clone(), local_cols.rd_rs2_ptr)
            - load_shift_amount;

        self.memory_bridge
            .read(
                MemoryAddress::new(read_as, read_ptr),
                ctx.reads.1,
                timestamp_pp(),
                &local_cols.read_data_aux,
            )
            .eval(builder, is_valid.clone());

        let write_aux_cols = MemoryWriteAuxCols::from_base(local_cols.write_base_aux, ctx.reads.0);

        // write_as is 1 for loads and [local_cols.mem_as] for stores
        let write_as = select::<AB::Expr>(
            is_load.clone(),
            AB::F::from_canonical_u32(RV32_REGISTER_AS),
            local_cols.mem_as,
        );

        // write_ptr is rd_rs2_ptr for loads and mem_ptr for stores
        let write_ptr = select::<AB::Expr>(is_load.clone(), local_cols.rd_rs2_ptr, mem_ptr.clone())
            - store_shift_amount;

        self.memory_bridge
            .write(
                MemoryAddress::new(write_as, write_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &write_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_rs2_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.mem_as.into(),
                    local_cols.needs_write.into(),
                    local_cols.imm_sign.into(),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &Rv32LoadStoreAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32LoadStoreAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs1_ptr: u32,
    pub rs1_val: u32,
    pub rs1_aux_record: MemoryReadAuxRecord,

    pub rd_rs2_ptr: u32,
    pub read_data_aux: MemoryReadAuxRecord,
    pub imm: u16,
    pub imm_sign: bool,

    pub mem_as: u8,

    pub write_prev_timestamp: u32,
}

/// This chip reads rs1 and gets a intermediate memory pointer address with rs1 + imm.
/// In case of Loads, reads from the shifted intermediate pointer and writes to rd.
/// In case of Stores, reads from rs2 and writes to the shifted intermediate pointer.
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32LoadStoreAdapterExecutor {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv32LoadStoreAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F> AdapterTraceExecutor<F> for Rv32LoadStoreAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32LoadStoreAdapterCols<u8>>();
    type ReadData = (
        (
            [u32; RV32_REGISTER_NUM_LIMBS],
            [u8; RV32_REGISTER_NUM_LIMBS],
        ),
        u8,
    );
    type WriteData = [u32; RV32_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv32LoadStoreAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            g,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        record.rs1_ptr = b.as_canonical_u32();
        record.rs1_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.rs1_ptr,
            &mut record.rs1_aux_record.prev_timestamp,
        ));

        record.imm = c.as_canonical_u32() as u16;
        record.imm_sign = g.is_one();
        let imm_extended = record.imm as u32 + record.imm_sign as u32 * 0xffff0000;

        let ptr_val = record.rs1_val.wrapping_add(imm_extended);
        let shift_amount = ptr_val & 3;
        let ptr_val = ptr_val - shift_amount;

        assert!(
            ptr_val < (1 << self.pointer_max_bits),
            "ptr_val: {ptr_val} = rs1_val: {} + imm_extended: {imm_extended} >= 2 ** {}",
            record.rs1_val,
            self.pointer_max_bits
        );

        // prev_data: We need to keep values of some cells to keep them unchanged when writing to
        // those cells
        let (read_data, prev_data) = match local_opcode {
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                debug_assert_eq!(e, F::from_canonical_u32(RV32_MEMORY_AS));
                record.mem_as = RV32_MEMORY_AS as u8;
                let read_data = tracing_read(
                    memory,
                    RV32_MEMORY_AS,
                    ptr_val,
                    &mut record.read_data_aux.prev_timestamp,
                );
                let prev_data = memory_read(memory.data(), RV32_REGISTER_AS, a.as_canonical_u32())
                    .map(u32::from);
                (read_data, prev_data)
            }
            STOREW | STOREH | STOREB => {
                let e = e.as_canonical_u32();
                debug_assert_ne!(e, RV32_IMM_AS);
                debug_assert_ne!(e, RV32_REGISTER_AS);
                record.mem_as = e as u8;
                let read_data = tracing_read(
                    memory,
                    RV32_REGISTER_AS,
                    a.as_canonical_u32(),
                    &mut record.read_data_aux.prev_timestamp,
                );
                let prev_data = if e == NATIVE_AS {
                    memory_read_native(memory.data(), ptr_val).map(|x: F| x.as_canonical_u32())
                } else {
                    memory_read(memory.data(), e, ptr_val).map(u32::from)
                };
                (read_data, prev_data)
            }
        };

        ((prev_data, read_data), shift_amount as u8)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction {
            opcode,
            a,
            d,
            e,
            f: enabled,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_ne!(e.as_canonical_u32(), RV32_IMM_AS);
        debug_assert_ne!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        if enabled != F::ZERO {
            record.rd_rs2_ptr = a.as_canonical_u32();

            record.write_prev_timestamp = match local_opcode {
                STOREW | STOREH | STOREB => {
                    let imm_extended = record.imm as u32 + record.imm_sign as u32 * 0xffff0000;
                    let ptr = record.rs1_val.wrapping_add(imm_extended) & !3;

                    if record.mem_as == 4 {
                        timed_write_native(memory, ptr, data.map(F::from_canonical_u32)).0
                    } else {
                        timed_write(memory, record.mem_as as u32, ptr, data.map(|x| x as u8)).0
                    }
                }
                LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                    timed_write(
                        memory,
                        RV32_REGISTER_AS,
                        record.rd_rs2_ptr,
                        data.map(|x| x as u8),
                    )
                    .0
                }
            };
        } else {
            record.rd_rs2_ptr = u32::MAX;
            memory.increment_timestamp();
        };
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32LoadStoreAdapterFiller {
    const WIDTH: usize = size_of::<Rv32LoadStoreAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        debug_assert!(self.range_checker_chip.range_max_bits() >= 15);

        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv32LoadStoreAdapterRecord
        let record: &Rv32LoadStoreAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();

        let needs_write = record.rd_rs2_ptr != u32::MAX;
        // Writing in reverse order
        adapter_row.needs_write = F::from_bool(needs_write);

        if needs_write {
            mem_helper.fill(
                record.write_prev_timestamp,
                record.from_timestamp + 2,
                &mut adapter_row.write_base_aux,
            );
        } else {
            mem_helper.fill_zero(&mut adapter_row.write_base_aux);
        }

        adapter_row.mem_as = F::from_canonical_u8(record.mem_as);
        let ptr = record
            .rs1_val
            .wrapping_add(record.imm as u32 + record.imm_sign as u32 * 0xffff0000);

        let ptr_limbs = [ptr & 0xffff, ptr >> 16];
        self.range_checker_chip
            .add_count(ptr_limbs[0] >> 2, RV32_CELL_BITS * 2 - 2);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - 16);
        adapter_row.mem_ptr_limbs = ptr_limbs.map(F::from_canonical_u32);

        adapter_row.imm_sign = F::from_bool(record.imm_sign);
        adapter_row.imm = F::from_canonical_u16(record.imm);

        mem_helper.fill(
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1,
            adapter_row.read_data_aux.as_mut(),
        );
        adapter_row.rd_rs2_ptr = if record.rd_rs2_ptr != u32::MAX {
            F::from_canonical_u32(record.rd_rs2_ptr)
        } else {
            F::ZERO
        };

        mem_helper.fill(
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );

        adapter_row.rs1_data = record.rs1_val.to_le_bytes().map(F::from_canonical_u8);
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
