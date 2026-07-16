use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VmAdapterAir, VmAdapterInterface, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{
            MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord,
            MemoryWriteAuxInput,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    PUBLIC_VALUES_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{
    byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, expand_to_rv64_block, memory_read_u16,
    ptr_to_field_u16_limbs, ptr_to_u16_limbs, rv64_address_add_imm, rv64_register_pointer,
    sign_extend_imm16, timed_write_u16, tracing_read, tracing_read_u16, try_rv64_bytes_to_u32,
    RV64_PTR_BITS, RV64_PTR_U16_LIMBS, RV64_REGISTER_NUM_LIMBS, U16_BITS,
};

pub struct StoreInstruction<T> {
    /// Guaranteed boolean by the core selector; may be a degree-2 expression.
    pub is_valid: T,
    /// Absolute opcode number.
    pub opcode: T,
    /// Byte offset of the effective pointer inside the 8-byte memory block.
    pub shift_amount: T,
    /// Flag for accesses that span two consecutive memory blocks; guaranteed boolean by the core
    /// as a sum of mutually exclusive selector flags matching the selected shift and access
    /// width. May be a degree-2 expression.
    pub store_cross: T,
}

pub struct Rv64StoreMultiByteAdapterAirInterface<AB: InteractionBuilder>(PhantomData<AB>);

/// The previous contents of the two consecutive memory blocks (the second is used only when the
/// access crosses a block boundary), followed by the source register data. The previous contents
/// feed both write auxes, so the core's read-modify-write inputs and the offline checker's
/// receive-side data are the same expressions by construction.
impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr>
    for Rv64StoreMultiByteAdapterAirInterface<AB>
{
    type Reads = ([[AB::Expr; BLOCK_FE_WIDTH]; 2], [AB::Expr; BLOCK_FE_WIDTH]);
    type Writes = [[AB::Expr; BLOCK_FE_WIDTH]; 2];
    type ProcessedInstruction = StoreInstruction<AB::Expr>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64StoreMultiByteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    /// Low 32 bits of the rs1 register, packed as two u16 cells.
    pub rs1_data: [T; RV64_PTR_U16_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    /// Source register pointer.
    pub rs2_ptr: T,
    pub read_data_aux: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    /// Low limb of the effective pointer for constraining rs1 + sign_extend(imm).
    pub mem_ptr_low_limb: T,
    pub mem_as: T,
    /// Carry into the high pointer limb when advancing to the second block wraps the low u16 limb.
    pub mem_ptr_carry: T,
    /// Timestamp aux for the memory write; previous data is provided by the core chip.
    pub write_base_aux: MemoryBaseAuxCols<T>,
    /// Timestamp aux for the second block write; only used when the access crosses a block
    /// boundary. Previous data is provided by the core chip.
    pub write1_base_aux: MemoryBaseAuxCols<T>,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64StoreMultiByteAdapterCols<u8>)]
pub struct Rv64StoreMultiByteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv64StoreMultiByteAdapterAir {
    fn width(&self) -> usize {
        Rv64StoreMultiByteAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64StoreMultiByteAdapterAir {
    type Interface = Rv64StoreMultiByteAdapterAirInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv64StoreMultiByteAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;
        let shift_amount = ctx.instruction.shift_amount;
        let cross = ctx.instruction.store_cross;

        // Read rs1 as a low 32-bit pointer value; the upper register cells are zero on the bus.
        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&local_cols.rs1_data);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rs1_ptr),
                ),
                rs1_data,
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, is_valid.clone());

        // Constrain mem_ptr = rs1 + sign_extend(imm) as a 32-bit addition. The booleanity
        // checks hold unconditionally (dummy rows are all-zero except `mem_as`), which keeps
        // their degree low since `is_valid` may be a degree-2 expression.
        let inv = AB::F::from_u32(1u32 << U16_BITS).inverse();
        let low_carry =
            (local_cols.rs1_data[0] + local_cols.imm - local_cols.mem_ptr_low_limb) * inv;
        builder.assert_bool(low_carry.clone());
        builder.assert_bool(local_cols.imm_sign);
        let mem_ptr_hi = local_cols.rs1_data[1] + low_carry - local_cols.imm_sign;

        // Prevent mem_ptr overflow while allowing the adapter to write the containing 8-byte block.
        let block_bytes = AB::F::from_u32(RV64_REGISTER_NUM_LIMBS as u32);
        let aligned_limb0 = local_cols.mem_ptr_low_limb - shift_amount.clone();
        self.range_bus
            .range_check(
                // (limb[0] - shift_amount) / 8 < 2^13 => limb[0] - shift_amount < 2^16
                aligned_limb0.clone() * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(mem_ptr_hi.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        // When the access crosses, range check the next block address too. mem_ptr_carry claims
        // whether adding one block wraps the low limb at 2^16; the low-limb check forces it.
        builder.assert_bool(local_cols.mem_ptr_carry);
        self.range_bus
            .range_check(
                (aligned_limb0.clone() + block_bytes
                    - local_cols.mem_ptr_carry * AB::F::from_u32(1u32 << U16_BITS))
                    * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, cross.clone());
        // The high limb changes only when the low limb wraps; otherwise the effective pointer's
        // high-limb check already covers the next block.
        self.range_bus
            .range_check(
                mem_ptr_hi.clone() + local_cols.mem_ptr_carry,
                self.pointer_max_bits - U16_BITS,
            )
            .eval(builder, local_cols.mem_ptr_carry);

        let mem_ptr = local_cols.mem_ptr_low_limb + mem_ptr_hi * AB::F::from_u32(1u32 << U16_BITS);

        // Constrain stores to writable u16-celled address spaces.
        builder.assert_bool(local_cols.mem_as - AB::Expr::TWO);

        let (prev_data, read_data) = ctx.reads;
        let [prev_data0, prev_data1] = prev_data;
        let [write_data0, write_data1] = ctx.writes;

        // Read the source register data to be written into memory.
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rs2_ptr),
                ),
                read_data,
                timestamp_pp(),
                &local_cols.read_data_aux,
            )
            .eval(builder, is_valid.clone());

        // Write the memory block containing the effective store address. The core supplies
        // previous cell values for any bytes not overwritten by this store.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    local_cols.mem_as,
                    byte_ptr_to_u16_ptr::<AB>(mem_ptr.clone() - shift_amount.clone()),
                ),
                write_data0,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(&local_cols.write_base_aux, prev_data0),
            )
            .eval(builder, is_valid.clone());

        // Write the next block when the access crosses into it. The timestamp slot is consumed
        // either way so the instruction has a static timestamp layout.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    local_cols.mem_as,
                    byte_ptr_to_u16_ptr::<AB>(
                        mem_ptr - shift_amount + AB::F::from_u32(RV64_REGISTER_NUM_LIMBS as u32),
                    ),
                ),
                write_data1,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(&local_cols.write1_base_aux, prev_data1),
            )
            .eval(builder, cross);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rs2_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    local_cols.mem_as.into(),
                    is_valid.clone(),
                    local_cols.imm_sign.into(),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_usize(timestamp_delta),
                },
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &Rv64StoreMultiByteAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64StoreMultiByteAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rs1_val: u32,
    pub rs1_aux_record: MemoryReadAuxRecord,
    pub read_data_aux: MemoryReadAuxRecord,
    pub write_prev_timestamp: u32,
    /// Prev timestamp of the second block write; `u32::MAX` when the access does not cross a
    /// block boundary.
    pub write1_prev_timestamp: u32,
    pub imm: u16,
    pub rs1_ptr: u8,
    pub rs2_ptr: u8,
    pub imm_sign: bool,
    pub mem_as: u8,
}

impl Rv64StoreMultiByteAdapterRecord {
    pub(crate) fn effective_ptr(&self) -> u32 {
        let addr = rv64_address_add_imm(
            self.rs1_val,
            sign_extend_imm16(self.imm as u32, self.imm_sign as u32),
        );
        u32::try_from(addr).expect("effective address exceeds u32 range")
    }

    pub(crate) fn shift_amount(&self) -> usize {
        (self.effective_ptr() & (RV64_REGISTER_NUM_LIMBS as u32 - 1)) as usize
    }

    pub(crate) fn crosses(&self) -> bool {
        self.write1_prev_timestamp != u32::MAX
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64StoreMultiByteAdapterExecutor<const STORE_WIDTH: usize> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64StoreMultiByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, const STORE_WIDTH: usize> AdapterTraceExecutor<F>
    for Rv64StoreMultiByteAdapterExecutor<STORE_WIDTH>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv64StoreMultiByteAdapterCols<u8>>();
    type ReadData = (([[u16; BLOCK_FE_WIDTH]; 2], [u16; BLOCK_FE_WIDTH]), u8);
    type WriteData = [[u16; BLOCK_FE_WIDTH]; 2];
    type RecordMut<'a> = &'a mut Rv64StoreMultiByteAdapterRecord;

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
            a, b, c, d, e, g, ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        let mem_as = e.as_canonical_u32();
        debug_assert_ne!(mem_as, RV64_IMM_AS);
        debug_assert_ne!(mem_as, RV64_REGISTER_AS);
        debug_assert!(mem_as == RV64_MEMORY_AS || mem_as == PUBLIC_VALUES_AS);

        record.rs1_ptr = rv64_register_pointer(b.as_canonical_u32());
        let rs1_bytes = tracing_read(
            memory,
            RV64_REGISTER_AS,
            u32::from(record.rs1_ptr),
            &mut record.rs1_aux_record.prev_timestamp,
        );
        record.rs1_val = try_rv64_bytes_to_u32(rs1_bytes).expect("upper 4 bytes must be zero");

        record.imm = c.as_canonical_u32() as u16;
        record.imm_sign = g.is_one();
        record.mem_as = mem_as as u8;
        let addr = rv64_address_add_imm(
            record.rs1_val,
            sign_extend_imm16(record.imm as u32, record.imm_sign as u32),
        );
        let ptr = u32::try_from(addr)
            .ok()
            .filter(|&ptr| {
                self.pointer_max_bits >= RV64_PTR_BITS
                    || u64::from(ptr) < (1u64 << self.pointer_max_bits)
            })
            .expect("effective address exceeds implemented memory address space");
        let shift_amount = ptr & (RV64_REGISTER_NUM_LIMBS as u32 - 1);
        let aligned_ptr = ptr - shift_amount;

        record.rs2_ptr = rv64_register_pointer(a.as_canonical_u32());
        let read_data = tracing_read_u16(
            memory,
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(u32::from(record.rs2_ptr)),
            &mut record.read_data_aux.prev_timestamp,
        );
        let prev_data0 = memory_read_u16(
            memory.data(),
            mem_as,
            byte_ptr_to_u16_ptr_value(aligned_ptr),
        );
        let crosses = shift_amount as usize + STORE_WIDTH > RV64_REGISTER_NUM_LIMBS;
        let prev_data1 = if crosses {
            let block1_ptr = aligned_ptr + RV64_REGISTER_NUM_LIMBS as u32;
            assert!(
                self.pointer_max_bits >= RV64_PTR_BITS
                    || u64::from(block1_ptr) + RV64_REGISTER_NUM_LIMBS as u64
                        <= (1u64 << self.pointer_max_bits),
                "crossing access exceeds implemented memory address space"
            );
            memory_read_u16(memory.data(), mem_as, byte_ptr_to_u16_ptr_value(block1_ptr))
        } else {
            [0; BLOCK_FE_WIDTH]
        };

        (([prev_data0, prev_data1], read_data), shift_amount as u8)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let shift_amount = record.shift_amount();
        let ptr = record.effective_ptr() & !(RV64_REGISTER_NUM_LIMBS as u32 - 1);
        record.write_prev_timestamp = timed_write_u16(
            memory,
            record.mem_as as u32,
            byte_ptr_to_u16_ptr_value(ptr),
            data[0],
        )
        .0;
        // The second block's timestamp slot is consumed either way so the instruction has a
        // static timestamp layout.
        if shift_amount + STORE_WIDTH > RV64_REGISTER_NUM_LIMBS {
            record.write1_prev_timestamp = timed_write_u16(
                memory,
                record.mem_as as u32,
                byte_ptr_to_u16_ptr_value(ptr + RV64_REGISTER_NUM_LIMBS as u32),
                data[1],
            )
            .0;
        } else {
            record.write1_prev_timestamp = u32::MAX;
            memory.increment_timestamp();
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64StoreMultiByteAdapterFiller {
    const WIDTH: usize = size_of::<Rv64StoreMultiByteAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        debug_assert!(self.range_checker_chip.range_max_bits() >= 15);

        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation written by the
        //   executor
        // - get_record_from_slice correctly interprets the bytes as Rv64StoreMultiByteAdapterRecord
        let record: &Rv64StoreMultiByteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        // The record and columns share the same row buffer, so copy record data
        // before writing columns.
        let from_pc = record.from_pc;
        let from_timestamp = record.from_timestamp;
        let rs1_ptr = record.rs1_ptr;
        let rs1_val = record.rs1_val;
        let rs1_prev_timestamp = record.rs1_aux_record.prev_timestamp;
        let rs2_ptr = record.rs2_ptr;
        let read_data_prev_timestamp = record.read_data_aux.prev_timestamp;
        let imm = record.imm;
        let imm_sign = record.imm_sign;
        let mem_as = record.mem_as;
        let write_prev_timestamp = record.write_prev_timestamp;
        let write1_prev_timestamp = record.write1_prev_timestamp;
        let crosses = record.crosses();
        let ptr = record.effective_ptr();
        let shift_amount = record.shift_amount() as u32;
        let adapter_row: &mut Rv64StoreMultiByteAdapterCols<F> = adapter_row.borrow_mut();

        if crosses {
            mem_helper.fill(
                write1_prev_timestamp,
                from_timestamp + 3,
                &mut adapter_row.write1_base_aux,
            );
        } else {
            mem_helper.fill_zero(&mut adapter_row.write1_base_aux);
        }
        mem_helper.fill(
            write_prev_timestamp,
            from_timestamp + 2,
            &mut adapter_row.write_base_aux,
        );

        adapter_row.mem_as = F::from_u8(mem_as);
        let ptr_limbs = ptr_to_u16_limbs(ptr).map(u32::from);
        let aligned_limb0 = ptr_limbs[0] - shift_amount;
        self.range_checker_chip
            .add_count(aligned_limb0 >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);

        let next_block_low_sum = aligned_limb0 + RV64_REGISTER_NUM_LIMBS as u32;
        let carry = crosses && next_block_low_sum == 1 << U16_BITS;
        adapter_row.mem_ptr_carry = F::from_bool(carry);
        if crosses {
            self.range_checker_chip.add_count(
                (next_block_low_sum - ((carry as u32) << U16_BITS)) >> 3,
                U16_BITS - 3,
            );
        }
        if carry {
            self.range_checker_chip.add_count(
                ptr_limbs[1] + carry as u32,
                self.pointer_max_bits - U16_BITS,
            );
        }

        adapter_row.imm_sign = F::from_bool(imm_sign);
        adapter_row.imm = F::from_u16(imm);
        adapter_row.rs2_ptr = F::from_u8(rs2_ptr);

        mem_helper.fill(
            read_data_prev_timestamp,
            from_timestamp + 1,
            adapter_row.read_data_aux.as_mut(),
        );
        mem_helper.fill(
            rs1_prev_timestamp,
            from_timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );

        adapter_row.rs1_data = ptr_to_field_u16_limbs(rs1_val);
        adapter_row.rs1_ptr = F::from_u8(rs1_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);
    }
}
