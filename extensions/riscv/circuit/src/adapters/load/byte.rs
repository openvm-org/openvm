use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, Postflight, PostflightError, PostflightStep, VmAdapterAir,
        VmAdapterInterface, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
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
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{
    byte_ptr_to_u16_ptr, byte_ptr_to_u16_ptr_value, checked_byte_ptr_to_u16_ptr_value,
    expand_to_rv64_block, memory_read_u16, ptr_to_field_u16_limbs, ptr_to_u16_limbs,
    rv64_address_add_imm, rv64_register_pointer, sign_extend_imm16, timed_write_u16, tracing_read,
    tracing_read_u16, try_rv64_bytes_to_u32, RV64_PTR_BITS, RV64_PTR_U16_LIMBS,
    RV64_REGISTER_NUM_LIMBS, U16_BITS,
};

// Byte loads never cross a memory block, so this adapter has no second-block columns.

pub struct LoadByteInstruction<T> {
    /// Boolean flag constrained by the core indicating whether this row is active.
    pub is_valid: T,
    /// Absolute opcode number.
    pub opcode: T,
    /// Byte offset of the effective pointer inside the 8-byte memory block.
    pub shift_amount: T,
}

pub struct Rv64LoadByteAdapterAirInterface;

impl<T> VmAdapterInterface<T> for Rv64LoadByteAdapterAirInterface {
    type Reads = [T; BLOCK_FE_WIDTH];
    type Writes = [T; BLOCK_FE_WIDTH];
    type ProcessedInstruction = LoadByteInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64LoadByteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    /// Low 32 bits of the rs1 register, packed as two u16 cells.
    pub rs1_data: [T; RV64_PTR_U16_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    /// Destination register pointer.
    pub rd_ptr: T,
    pub read_data_aux: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    /// Low limb of the effective pointer for constraining rs1 + sign_extend(imm).
    pub mem_ptr_low_limb: T,
    pub write_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
    /// Only writes to rd if the load is valid and rd is not x0.
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64LoadByteAdapterCols<u8>)]
pub struct Rv64LoadByteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv64LoadByteAdapterAir {
    fn width(&self) -> usize {
        Rv64LoadByteAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64LoadByteAdapterAir {
    type Interface = Rv64LoadByteAdapterAirInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv64LoadByteAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;
        let shift_amount = ctx.instruction.shift_amount;
        let write_count = local_cols.needs_write;

        // This constraint ensures that the register write only occurs when `is_valid == 1`.
        builder.assert_bool(write_count);
        builder.when(write_count).assert_one(is_valid.clone());
        // If a valid load does not write, then it must target x0.
        builder
            .when(is_valid.clone() - write_count)
            .assert_zero(local_cols.rd_ptr);

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

        // Constrain mem_ptr = rs1 + sign_extend(imm) as a 32-bit addition.
        let inv = AB::F::from_u32(1u32 << U16_BITS).inverse();
        let low_carry =
            (local_cols.rs1_data[0] + local_cols.imm - local_cols.mem_ptr_low_limb) * inv;
        builder.assert_bool(low_carry.clone());

        builder.assert_bool(local_cols.imm_sign);
        let mem_ptr_hi = local_cols.rs1_data[1] + low_carry - local_cols.imm_sign;

        // Prevent mem_ptr overflow while allowing the adapter to read the containing 8-byte block.
        let block_bytes = AB::F::from_u32(MEMORY_BLOCK_BYTES as u32);
        let aligned_limb = local_cols.mem_ptr_low_limb - shift_amount.clone();
        self.range_bus
            .range_check(
                // aligned_limb / 8 < 2^13 => aligned_limb < 2^16
                aligned_limb * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(mem_ptr_hi.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        let mem_ptr = local_cols.mem_ptr_low_limb + mem_ptr_hi * AB::F::from_u32(1u32 << U16_BITS);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_MEMORY_AS),
                    byte_ptr_to_u16_ptr::<AB>(mem_ptr - shift_amount),
                ),
                ctx.reads,
                timestamp_pp(),
                &local_cols.read_data_aux,
            )
            .eval(builder, is_valid.clone());

        // Write the loaded value into rd, unless rd is x0.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rd_ptr),
                ),
                ctx.writes.clone(),
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_MEMORY_AS),
                    local_cols.needs_write.into(),
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
        let local_cols: &Rv64LoadByteAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64LoadByteAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rs1_val: u32,
    pub rs1_aux_record: MemoryReadAuxRecord,
    pub read_data_aux: MemoryReadAuxRecord,
    pub imm: u16,
    pub imm_sign: bool,
    pub write_prev_timestamp: u32,
    pub write_prev_data: [u16; BLOCK_FE_WIDTH],
    pub rs1_ptr: u8,
    /// `u8::MAX` when the load does not write a register.
    pub rd_ptr: u8,
}

impl Rv64LoadByteAdapterRecord {
    pub(crate) fn effective_ptr(&self) -> u32 {
        let addr = rv64_address_add_imm(
            self.rs1_val,
            sign_extend_imm16(self.imm as u32, self.imm_sign as u32),
        );
        u32::try_from(addr).expect("effective address exceeds u32 range")
    }

    pub(crate) fn shift_amount(&self) -> usize {
        (self.effective_ptr() & (MEMORY_BLOCK_BYTES as u32 - 1)) as usize
    }
}

/// Reads rs1, computes the effective memory pointer, reads the containing memory block, and writes
/// the loaded value to rd.
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64LoadByteAdapterExecutor {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64LoadByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F> AdapterTraceExecutor<F> for Rv64LoadByteAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv64LoadByteAdapterCols<u8>>();
    type ReadData = (([u16; BLOCK_FE_WIDTH], [u16; BLOCK_FE_WIDTH]), u8);
    type WriteData = [u16; BLOCK_FE_WIDTH];
    type RecordMut<'a> = &'a mut Rv64LoadByteAdapterRecord;

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
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

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
        let shift_amount = ptr & (MEMORY_BLOCK_BYTES as u32 - 1);
        let aligned_ptr = ptr - shift_amount;

        let read_data = tracing_read_u16(
            memory,
            RV64_MEMORY_AS,
            byte_ptr_to_u16_ptr_value(aligned_ptr),
            &mut record.read_data_aux.prev_timestamp,
        );
        let prev_data = memory_read_u16(
            memory.data(),
            RV64_REGISTER_AS,
            byte_ptr_to_u16_ptr_value(a.as_canonical_u32()),
        );
        record.write_prev_data = prev_data;

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
        let &Instruction { a, f: enabled, .. } = instruction;
        if enabled != F::ZERO {
            record.rd_ptr = rv64_register_pointer(a.as_canonical_u32());
            record.write_prev_timestamp = timed_write_u16(
                memory,
                RV64_REGISTER_AS,
                byte_ptr_to_u16_ptr_value(u32::from(record.rd_ptr)),
                data,
            )
            .0;
        } else {
            record.rd_ptr = u8::MAX;
            memory.increment_timestamp();
        };
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv64LoadByteAdapterFiller {
    const WIDTH: usize = size_of::<Rv64LoadByteAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        debug_assert!(self.range_checker_chip.range_max_bits() >= 15);

        // SAFETY:
        // - the executor wrote an `Rv64LoadByteAdapterRecord` into this row buffer
        // - the record fits within the byte adapter's column layout
        // - `get_record_from_slice` returns the record representation at the start of the row
        let record: &Rv64LoadByteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        // Copy the record before reusing its row buffer for columns.
        let from_pc = record.from_pc;
        let from_timestamp = record.from_timestamp;
        let rs1_ptr = record.rs1_ptr;
        let rs1_val = record.rs1_val;
        let rs1_prev_timestamp = record.rs1_aux_record.prev_timestamp;
        let rd_ptr = record.rd_ptr;
        let read_data_prev_timestamp = record.read_data_aux.prev_timestamp;
        let imm = record.imm;
        let imm_sign = record.imm_sign;
        let write_prev_timestamp = record.write_prev_timestamp;
        let write_prev_data = record.write_prev_data;
        let ptr = record.effective_ptr();
        let shift_amount = record.shift_amount() as u32;
        let adapter_row: &mut Rv64LoadByteAdapterCols<F> = adapter_row.borrow_mut();

        let needs_write = rd_ptr != u8::MAX;
        adapter_row.needs_write = F::from_bool(needs_write);
        adapter_row.rd_ptr = if needs_write {
            F::from_u8(rd_ptr)
        } else {
            F::ZERO
        };

        if needs_write {
            mem_helper.fill(
                write_prev_timestamp,
                from_timestamp + 2,
                &mut adapter_row.write_aux.base,
            );
            adapter_row.write_aux.prev_data = write_prev_data.map(F::from_u16);
        } else {
            mem_helper.fill_zero(&mut adapter_row.write_aux.base);
            adapter_row.write_aux.prev_data = [F::ZERO; BLOCK_FE_WIDTH];
        }

        let ptr_limbs = ptr_to_u16_limbs(ptr).map(u32::from);
        self.range_checker_chip
            .add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);

        adapter_row.imm_sign = F::from_bool(imm_sign);
        adapter_row.imm = F::from_u16(imm);

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

impl Rv64LoadByteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64LoadByteAdapterCols<F>,
        compute: impl FnOnce([u16; BLOCK_FE_WIDTH], usize) -> [u16; BLOCK_FE_WIDTH],
    ) -> Result<([u16; BLOCK_FE_WIDTH], usize, [u16; BLOCK_FE_WIDTH]), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
        {
            return Err(PostflightError::new(
                "byte-load instruction has invalid address spaces",
            ));
        }
        let needs_write = match instruction.f.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "byte-load instruction has a non-boolean write enable",
                ));
            }
        };
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "byte-load instruction has a non-boolean immediate sign",
                ));
            }
        };
        let imm = instruction.c.as_canonical_u32();
        if imm > u16::MAX as u32 {
            return Err(PostflightError::new(
                "byte-load instruction has a non-canonical immediate",
            ));
        }

        let rs1_ptr = instruction.b.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        if rs1_ptr > u32::from(u8::MAX) || !rs1_ptr.is_multiple_of(RV64_REGISTER_NUM_LIMBS as u32) {
            return Err(PostflightError::new(
                "byte-load instruction has an invalid source register pointer",
            ));
        }
        if rd_ptr > u32::from(u8::MAX) || !rd_ptr.is_multiple_of(RV64_REGISTER_NUM_LIMBS as u32) {
            return Err(PostflightError::new(
                "byte-load instruction has an invalid destination register pointer",
            ));
        }
        if needs_write != (rd_ptr != 0) {
            return Err(PostflightError::new(
                "byte-load write enable does not match its destination register",
            ));
        }

        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs1_ptr)?;
        let rd_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        if rs1.value[RV64_PTR_U16_LIMBS..].iter().any(|&x| x != 0) {
            return Err(PostflightError::new(
                "byte-load base register exceeds the implemented pointer width",
            ));
        }
        let rs1_val = u32::from(rs1.value[0]) | (u32::from(rs1.value[1]) << U16_BITS);
        let effective_ptr = u32::try_from(rv64_address_add_imm(
            rs1_val,
            sign_extend_imm16(imm, u32::from(imm_sign)),
        ))
        .map_err(|_| PostflightError::new("byte-load effective address exceeds u32"))?;
        if self.pointer_max_bits < u32::BITS as usize
            && u64::from(effective_ptr) >= (1u64 << self.pointer_max_bits)
        {
            return Err(PostflightError::new(
                "byte-load effective address exceeds implemented memory",
            ));
        }
        let shift_amount = effective_ptr as usize & (MEMORY_BLOCK_BYTES - 1);
        let aligned_ptr = effective_ptr - shift_amount as u32;
        let memory = replay.read_u16(
            RV64_MEMORY_AS,
            checked_byte_ptr_to_u16_ptr_value(aligned_ptr)?,
        )?;
        let output = compute(memory.value, shift_amount);

        adapter_row.needs_write = F::from_bool(needs_write);
        if needs_write {
            let write = replay.write_u16(RV64_REGISTER_AS, rd_u16_ptr, output)?;
            mem_helper.fill(
                write.previous_timestamp,
                write.timestamp,
                &mut adapter_row.write_aux.base,
            );
            adapter_row.write_aux.prev_data = write.previous_value.map(F::from_u16);
            adapter_row.rd_ptr = F::from_u32(rd_ptr);
        } else {
            replay.advance_timestamp(1)?;
            mem_helper.fill_zero(&mut adapter_row.write_aux.base);
            adapter_row.write_aux.prev_data = [F::ZERO; BLOCK_FE_WIDTH];
            adapter_row.rd_ptr = F::ZERO;
        }
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        let ptr_limbs = ptr_to_u16_limbs(effective_ptr).map(u32::from);
        self.range_checker_chip
            .add_count((ptr_limbs[0] - shift_amount as u32) >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);
        adapter_row.imm_sign = F::from_bool(imm_sign);
        adapter_row.imm = F::from_u32(imm);
        mem_helper.fill(
            memory.previous_timestamp,
            memory.timestamp,
            adapter_row.read_data_aux.as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );
        adapter_row.rs1_data = ptr_to_field_u16_limbs(rs1_val);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok((memory.value, shift_amount, output))
    }
}
