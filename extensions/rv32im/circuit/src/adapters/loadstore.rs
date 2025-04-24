use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, ExecutionBridge, ExecutionState,
        VmAdapterAir, VmAdapterInterface,
    },
    system::memory::{
        offline_checker::{MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory, RecordId,
    },
};
use openvm_circuit_primitives::{
    utils::{not, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{tracing_read_reg, tracing_write_reg, RV32_CELL_BITS};

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

/// The LoadStoreAdapter separates Runtime and Air AdapterInterfaces.
/// This is necessary because `prev_data` should be owned by the core chip and sent to the adapter,
/// and it must have an AB::Var type in AIR as to satisfy the memory_bridge interface.
/// This is achieved by having different types for reads and writes in Air AdapterInterface.
/// This method ensures that there are no modifications to the global interfaces.
///
/// Here 2 reads represent read_data and prev_data,
/// The second element of the tuple in Reads is the shift amount needed to be passed to the core
/// chip Getting the intermediate pointer is completely internal to the adapter and shouldn't be a
/// part of the AdapterInterface
pub struct Rv32LoadStoreAdapterRuntimeInterface<T>(PhantomData<T>);
impl<T> VmAdapterInterface<T> for Rv32LoadStoreAdapterRuntimeInterface<T> {
    type Reads = ([[T; RV32_REGISTER_NUM_LIMBS]; 2], T);
    type Writes = [[T; RV32_REGISTER_NUM_LIMBS]; 1];
    type ProcessedInstruction = ();
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Rv32LoadStoreReadRecord<F: Field> {
    pub rs1_record: RecordId,
    /// This will be a read from a register in case of Stores and a read from RISC-V memory in case
    /// of Loads.
    pub read: RecordId,
    pub rs1_ptr: F,
    pub imm: F,
    pub imm_sign: F,
    pub mem_as: F,
    pub mem_ptr_limbs: [u32; 2],
    pub shift_amount: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Rv32LoadStoreWriteRecord<F: Field> {
    /// This will be a write to a register in case of Load and a write to RISC-V memory in case of
    /// Stores. For better struct packing, `RecordId(usize::MAX)` is used to indicate that
    /// there is no write.
    pub write_id: RecordId,
    pub from_state: ExecutionState<u32>,
    pub rd_rs2_ptr: F,
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

/// This chip reads rs1 and gets a intermediate memory pointer address with rs1 + imm.
/// In case of Loads, reads from the shifted intermediate pointer and writes to rd.
/// In case of Stores, reads from rs2 and writes to the shifted intermediate pointer.
pub struct Rv32LoadStoreAdapterStep {
    pointer_max_bits: usize,
    // TODO(ayush): move to trace generation context
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl Rv32LoadStoreAdapterStep {
    pub fn new(
        pointer_max_bits: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 15);
        Self {
            pointer_max_bits,
            range_checker_chip,
        }
    }
}

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32LoadStoreAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32LoadStoreAdapterCols<u8>>();
    type ReadData = (
        ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]),
        u32,
    );
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction {
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
        debug_assert!(e.as_canonical_u32() != RV32_IMM_AS);

        let adapter_row: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let rs1 = tracing_read_reg(
            memory,
            b.as_canonical_u32(),
            (&mut adapter_row.rs1_ptr, &mut adapter_row.rs1_aux_cols),
        );

        let rs1_val = u32::from_le_bytes(rs1);
        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        let shift_amount = ptr_val % 4;
        assert!(
            ptr_val < (1 << self.pointer_max_bits),
            "ptr_val: {ptr_val} = rs1_val: {rs1_val} + imm_extended: {imm_extended} >= 2 ** {}",
            self.pointer_max_bits
        );

        let mem_ptr_limbs = array::from_fn(|i| ((ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff));

        let ptr_val = ptr_val - shift_amount;
        let read_data = match local_opcode {
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                // memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(e, F::from_canonical_u32(ptr_val))
                // TODO(ayush): this should be tracing_read_mem. check other columns
                tracing_read_reg(
                    memory,
                    ptr_val,
                    (&mut adapter_row.rs1_ptr, &mut adapter_row.read_data_aux),
                )
            }
            STOREW | STOREH | STOREB => {
                // memory.read(d.as_canonical_u32(), a.as_canonical_u32())
                // TODO(ayush): what should be rs1_ptr here?
                tracing_read_reg(
                    memory,
                    a.as_canonical_u32(),
                    (&mut adapter_row.rs1_ptr, &mut adapter_row.read_data_aux),
                )
            }
        };

        // We need to keep values of some cells to keep them unchanged when writing to those cells
        let prev_data = match local_opcode {
            STOREW | STOREH | STOREB => {
                // array::from_fn(|i| {
                //     memory.unsafe_read_cell::<u8>(e, F::from_canonical_usize(ptr_val as usize + i))
                // })
                // TODO(ayush): this probably doesn't need to be traced
                // memory.read(e.as_canonical_u32(), ptr_val);
                tracing_read_reg(
                    memory,
                    ptr_val,
                    (&mut adapter_row.rs1_ptr, &mut adapter_row.read_data_aux),
                )
            }
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                // array::from_fn(|i| memory.unsafe_read_cell::<u8>(d, a + F::from_canonical_usize(i)))
                // TODO(ayush): this probably doesn't need to be traced
                // memory.read(d.as_canonical_u32(), a.as_canonical_u32())
                tracing_read_reg(
                    memory,
                    a.as_canonical_u32(),
                    (&mut adapter_row.rs1_ptr, &mut adapter_row.read_data_aux),
                )
            }
        };

        adapter_row
            .rs1_data
            .copy_from_slice(&rs1.map(F::from_canonical_u8));
        adapter_row.imm = *c;
        adapter_row.imm_sign = *g;
        adapter_row.mem_ptr_limbs = mem_ptr_limbs.map(F::from_canonical_u32);
        adapter_row.mem_as = *e;

        ((prev_data, read_data), shift_amount)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f: enabled,
            g,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(e.as_canonical_u32() != RV32_IMM_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let adapter_row: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();

        let rs1 = adapter_row.rs1_data.map(|x| x.as_canonical_u32() as u8);

        let rs1_val = u32::from_le_bytes(rs1);
        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        let shift_amount = ptr_val % 4;
        assert!(
            ptr_val < (1 << self.pointer_max_bits),
            "ptr_val: {ptr_val} = rs1_val: {rs1_val} + imm_extended: {imm_extended} >= 2 ** {}",
            self.pointer_max_bits
        );

        let mem_ptr_limbs: [u32; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|i| ((ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff));

        if *enabled != F::ZERO {
            adapter_row.needs_write = F::ONE;

            match local_opcode {
                STOREW | STOREH | STOREB => {
                    let ptr = mem_ptr_limbs[0] + mem_ptr_limbs[1] * (1 << (RV32_CELL_BITS * 2));
                    // TODO(ayush): write_aux is not complete. previous data is in the core columns
                    // memory.write(
                    //     e,
                    //     F::from_canonical_u32(ptr & 0xfffffffc),
                    //     &tmp_convert_to_u8s(output.writes[0]),
                    // )
                    tracing_write_reg(
                        memory,
                        a.as_canonical_u32(),
                        data,
                        (&mut adapter_row.rd_rs2_ptr, &mut adapter_row.write_base_aux),
                    );
                }
                LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                    // TODO(ayush): write_aux is not complete. previous data is in the core columns
                    // memory.write(d, a, &tmp_convert_to_u8s(output.writes[0]))
                    tracing_write_reg(
                        memory,
                        a.as_canonical_u32(),
                        data,
                        (&mut adapter_row.rd_rs2_ptr, &mut adapter_row.write_base_aux),
                    );
                }
            };
        } else {
            memory.increment_timestamp();
        };
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();

        let rs1 = adapter_row.rs1_data.map(|x| x.as_canonical_u32() as u8);
        let rs1_val = u32::from_le_bytes(rs1);

        let imm = adapter_row.imm.as_canonical_u32();
        let imm_sign = adapter_row.imm_sign.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        let shift_amount = ptr_val % 4;
        self.range_checker_chip.add_count(
            (adapter_row.mem_ptr_limbs[0].as_canonical_u32() - shift_amount) / 4,
            RV32_CELL_BITS * 2 - 2,
        );
        self.range_checker_chip.add_count(
            adapter_row.mem_ptr_limbs[1].as_canonical_u32(),
            self.pointer_max_bits - RV32_CELL_BITS * 2,
        );
    }
}

impl<Mem, F> AdapterExecutorE1<Mem, F> for Rv32LoadStoreAdapterStep
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = (
        ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]),
        u32,
    );
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    fn read(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction {
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
        debug_assert!(e.as_canonical_u32() != RV32_IMM_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { memory.read(d.as_canonical_u32(), b.as_canonical_u32()) };
        let rs1_val = u32::from_le_bytes(rs1_bytes);

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        assert!(
            ptr_val < (1 << self.pointer_max_bits),
            "ptr_val: {ptr_val} = rs1_val: {rs1_val} + imm_extended: {imm_extended} >= 2 ** {}",
            self.pointer_max_bits
        );
        let shift_amount = ptr_val % 4;

        let ptr_val = ptr_val - shift_amount; // aligned ptr

        let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = match local_opcode {
            LOADW | LOADB | LOADH | LOADBU | LOADHU => unsafe {
                memory.read(e.as_canonical_u32(), ptr_val)
            },
            STOREW | STOREH | STOREB => unsafe {
                memory.read(d.as_canonical_u32(), a.as_canonical_u32())
            },
        };

        // For stores, we need the previous memory content to preserve unchanged bytes
        let prev_data: [u8; RV32_REGISTER_NUM_LIMBS] = match local_opcode {
            STOREW | STOREH | STOREB => unsafe { memory.read(e.as_canonical_u32(), ptr_val) },
            LOADW | LOADB | LOADH | LOADBU | LOADHU => unsafe {
                memory.read(d.as_canonical_u32(), a.as_canonical_u32())
            },
        };

        ((prev_data, read_data), shift_amount)
    }

    fn write(&self, memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData) {
        // TODO(ayush): remove duplication with read
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f: enabled,
            g,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(e.as_canonical_u32() != RV32_IMM_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { memory.read(d.as_canonical_u32(), b.as_canonical_u32()) };
        let rs1_val = u32::from_le_bytes(rs1_bytes);

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        assert!(
            ptr_val < (1 << self.pointer_max_bits),
            "ptr_val: {ptr_val} = rs1_val: {rs1_val} + imm_extended: {imm_extended} >= 2 ** {}",
            self.pointer_max_bits
        );
        let shift_amount = ptr_val % 4;

        let ptr_val = ptr_val - shift_amount; // aligned ptr

        let mem_ptr_limbs: [u32; 2] =
            array::from_fn(|i| ((ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff));

        if *enabled != F::ZERO {
            match local_opcode {
                STOREW | STOREH | STOREB => {
                    let ptr = mem_ptr_limbs[0] + mem_ptr_limbs[1] * (1 << (RV32_CELL_BITS * 2));
                    unsafe { memory.write(e.as_canonical_u32(), ptr & 0xfffffffc, &data) };
                }
                LOADW | LOADB | LOADH | LOADBU | LOADHU => unsafe {
                    memory.write(d.as_canonical_u32(), a.as_canonical_u32(), &data);
                },
            }
        }
    }
}

// impl<F: PrimeField32> VmAdapterChip<F> for Rv32LoadStoreAdapterChip<F> {
//     type ReadRecord = Rv32LoadStoreReadRecord<F>;
//     type WriteRecord = Rv32LoadStoreWriteRecord<F>;
//     type Air = Rv32LoadStoreAdapterAir;
//     type Interface = Rv32LoadStoreAdapterRuntimeInterface<F>;

//     #[allow(clippy::type_complexity)]
//     fn preprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//     ) -> Result<(
//         <Self::Interface as VmAdapterInterface<F>>::Reads,
//         Self::ReadRecord,
//     )> {
//         let Instruction {
//             opcode,
//             a,
//             b,
//             c,
//             d,
//             e,
//             g,
//             ..
//         } = *instruction;
//         debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
//         debug_assert!(e.as_canonical_u32() != RV32_IMM_AS);

//         let local_opcode = Rv32LoadStoreOpcode::from_usize(
//             opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
//         );
//         let rs1_record = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(d, b);

//         let rs1_val = u32::from_le_bytes(rs1_record.1);
//         let imm = c.as_canonical_u32();
//         let imm_sign = g.as_canonical_u32();
//         let imm_extended = imm + imm_sign * 0xffff0000;

//         let ptr_val = rs1_val.wrapping_add(imm_extended);
//         let shift_amount = ptr_val % 4;
//         assert!(
//             ptr_val < (1 << self.air.pointer_max_bits),
//             "ptr_val: {ptr_val} = rs1_val: {rs1_val} + imm_extended: {imm_extended} >= 2 ** {}",
//             self.air.pointer_max_bits
//         );

//         let mem_ptr_limbs = array::from_fn(|i| ((ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff));

//         let ptr_val = ptr_val - shift_amount;
//         let read_record = match local_opcode {
//             LOADW | LOADB | LOADH | LOADBU | LOADHU => {
//                 memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(e, F::from_canonical_u32(ptr_val))
//             }
//             STOREW | STOREH | STOREB => memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(d, a),
//         };

//         // We need to keep values of some cells to keep them unchanged when writing to those cells
//         let prev_data = match local_opcode {
//             STOREW | STOREH | STOREB => array::from_fn(|i| {
//                 memory.unsafe_read_cell::<u8>(e, F::from_canonical_usize(ptr_val as usize + i))
//             }),
//             LOADW | LOADB | LOADH | LOADBU | LOADHU => {
//                 array::from_fn(|i| memory.unsafe_read_cell::<u8>(d, a + F::from_canonical_usize(i)))
//             }
//         };

//         Ok((
//             (
//                 [
//                     prev_data.map(F::from_canonical_u8),
//                     read_record.1.map(F::from_canonical_u8),
//                 ],
//                 F::from_canonical_u32(shift_amount),
//             ),
//             Self::ReadRecord {
//                 rs1_record: rs1_record.0,
//                 rs1_ptr: b,
//                 read: read_record.0,
//                 imm: c,
//                 imm_sign: g,
//                 shift_amount,
//                 mem_ptr_limbs,
//                 mem_as: e,
//             },
//         ))
//     }

//     fn postprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//         from_state: ExecutionState<u32>,
//         output: AdapterRuntimeContext<F, Self::Interface>,
//         read_record: &Self::ReadRecord,
//     ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
//         let Instruction {
//             opcode,
//             a,
//             d,
//             e,
//             f: enabled,
//             ..
//         } = *instruction;

//         let local_opcode = Rv32LoadStoreOpcode::from_usize(
//             opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
//         );

//         let write_id = if enabled != F::ZERO {
//             let (record_id, _) = match local_opcode {
//                 STOREW | STOREH | STOREB => {
//                     let ptr = read_record.mem_ptr_limbs[0]
//                         + read_record.mem_ptr_limbs[1] * (1 << (RV32_CELL_BITS * 2));
//                     memory.write(
//                         e,
//                         F::from_canonical_u32(ptr & 0xfffffffc),
//                         &tmp_convert_to_u8s(output.writes[0]),
//                     )
//                 }
//                 LOADW | LOADB | LOADH | LOADBU | LOADHU => {
//                     memory.write(d, a, &tmp_convert_to_u8s(output.writes[0]))
//                 }
//             };
//             record_id
//         } else {
//             memory.increment_timestamp();
//             // RecordId will never get to usize::MAX, so it can be used as a flag for no write
//             RecordId(usize::MAX)
//         };

//         Ok((
//             ExecutionState {
//                 pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
//                 timestamp: memory.timestamp(),
//             },
//             Self::WriteRecord {
//                 from_state,
//                 write_id,
//                 rd_rs2_ptr: a,
//             },
//         ))
//     }

//     fn generate_trace_row(
//         &self,
//         row_slice: &mut [F],
//         read_record: Self::ReadRecord,
//         write_record: Self::WriteRecord,
//         memory: &OfflineMemory<F>,
//     ) {
//         self.range_checker_chip.add_count(
//             (read_record.mem_ptr_limbs[0] - read_record.shift_amount) / 4,
//             RV32_CELL_BITS * 2 - 2,
//         );
//         self.range_checker_chip.add_count(
//             read_record.mem_ptr_limbs[1],
//             self.air.pointer_max_bits - RV32_CELL_BITS * 2,
//         );

//         let aux_cols_factory = memory.aux_cols_factory();
//         let adapter_cols: &mut Rv32LoadStoreAdapterCols<_> = row_slice.borrow_mut();
//         adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
//         let rs1 = memory.record_by_id(read_record.rs1_record);
//         adapter_cols.rs1_data.copy_from_slice(rs1.data_slice());
//         aux_cols_factory.generate_read_aux(rs1, &mut adapter_cols.rs1_aux_cols);
//         adapter_cols.rs1_ptr = read_record.rs1_ptr;
//         adapter_cols.rd_rs2_ptr = write_record.rd_rs2_ptr;
//         let read = memory.record_by_id(read_record.read);
//         aux_cols_factory.generate_read_aux(read, &mut adapter_cols.read_data_aux);
//         adapter_cols.imm = read_record.imm;
//         adapter_cols.imm_sign = read_record.imm_sign;
//         adapter_cols.mem_ptr_limbs = read_record.mem_ptr_limbs.map(F::from_canonical_u32);
//         adapter_cols.mem_as = read_record.mem_as;
//         if write_record.write_id.0 != usize::MAX {
//             let write = memory.record_by_id(write_record.write_id);
//             aux_cols_factory.generate_base_aux(write, &mut adapter_cols.write_base_aux);
//             adapter_cols.needs_write = F::ONE;
//         }
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }
