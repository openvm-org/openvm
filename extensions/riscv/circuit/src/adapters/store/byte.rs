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
            MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxInput,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
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
    Rv64StoreMultiByteAdapterRecord, StoreInstruction, RV64_PTR_BITS, RV64_PTR_U16_LIMBS,
    RV64_REGISTER_NUM_LIMBS, U16_BITS,
};

// Byte stores never cross a memory block, so this adapter has no second-block columns.

pub struct Rv64StoreByteAdapterAirInterface<AB: InteractionBuilder>(PhantomData<AB>);

impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr> for Rv64StoreByteAdapterAirInterface<AB> {
    type Reads = ([AB::Expr; BLOCK_FE_WIDTH], [AB::Expr; BLOCK_FE_WIDTH]);
    type Writes = [[AB::Expr; BLOCK_FE_WIDTH]; 1];
    type ProcessedInstruction = StoreInstruction<AB::Expr>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64StoreByteAdapterCols<T> {
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
    /// Timestamp aux for the memory write; previous data is provided by the core chip.
    pub write_base_aux: MemoryBaseAuxCols<T>,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64StoreByteAdapterCols<u8>)]
pub struct Rv64StoreByteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv64StoreByteAdapterAir {
    fn width(&self) -> usize {
        Rv64StoreByteAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64StoreByteAdapterAir {
    type Interface = Rv64StoreByteAdapterAirInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv64StoreByteAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;
        let shift_amount = ctx.instruction.shift_amount;

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
        self.range_bus
            .range_check(
                // (limb[0] - shift_amount) / 8 < 2^13 => limb[0] - shift_amount < 2^16
                (local_cols.mem_ptr_low_limb - shift_amount.clone())
                    * AB::F::from_u32(RV64_REGISTER_NUM_LIMBS as u32).inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(mem_ptr_hi.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        let mem_ptr = local_cols.mem_ptr_low_limb + mem_ptr_hi * AB::F::from_u32(1u32 << U16_BITS);

        // Constrain stores to writable u16-celled address spaces.
        builder.assert_bool(local_cols.mem_as - AB::Expr::TWO);

        let (prev_data, read_data) = ctx.reads;
        let [write_data] = ctx.writes;

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
                    byte_ptr_to_u16_ptr::<AB>(mem_ptr - shift_amount),
                ),
                write_data,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(&local_cols.write_base_aux, prev_data),
            )
            .eval(builder, is_valid.clone());

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
        let local_cols: &Rv64StoreByteAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64StoreByteAdapterExecutor {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64StoreByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F> AdapterTraceExecutor<F> for Rv64StoreByteAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv64StoreByteAdapterCols<u8>>();
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
        // Mark the optional second write absent.
        record.write1_prev_timestamp = u32::MAX;

        (
            ([prev_data0, [0; BLOCK_FE_WIDTH]], read_data),
            shift_amount as u8,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let ptr = record.effective_ptr() & !(RV64_REGISTER_NUM_LIMBS as u32 - 1);
        record.write_prev_timestamp = timed_write_u16(
            memory,
            record.mem_as as u32,
            byte_ptr_to_u16_ptr_value(ptr),
            data[0],
        )
        .0;
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64StoreByteAdapterFiller {
    const WIDTH: usize = size_of::<Rv64StoreByteAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        debug_assert!(self.range_checker_chip.range_max_bits() >= 15);

        // SAFETY:
        // - the executor wrote an `Rv64StoreMultiByteAdapterRecord` into this row buffer
        // - the record fits within the byte adapter's column layout
        // - `get_record_from_slice` returns the record representation at the start of the row
        let record: &Rv64StoreMultiByteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
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
        let ptr = record.effective_ptr();
        let shift_amount = record.shift_amount() as u32;
        let adapter_row: &mut Rv64StoreByteAdapterCols<F> = adapter_row.borrow_mut();

        mem_helper.fill(
            write_prev_timestamp,
            from_timestamp + 2,
            &mut adapter_row.write_base_aux,
        );

        adapter_row.mem_as = F::from_u8(mem_as);
        let ptr_limbs = ptr_to_u16_limbs(ptr).map(u32::from);
        self.range_checker_chip
            .add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);

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
