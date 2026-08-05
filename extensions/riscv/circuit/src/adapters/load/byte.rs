use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, Postflight, PostflightError,
        PostflightStep, VmAdapterAir, VmAdapterInterface, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{pc_to_idx, DEFAULT_PC_STEP},
    riscv::{MEMORY_AS, REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{
    address_add_imm, byte_ptr_limbs_to_cell_ptr_limbs_value, cell_ptr_hi_bits,
    checked_byte_ptr_to_u16_ptr_value, checked_register_u16_pointer, expand_to_block,
    ptr_to_field_u16_limbs, ptr_to_u16_limbs, reg_byte_ptr_to_cell_ptr_limbs, sign_extend_imm16,
    PTR_U16_LIMBS, REGISTER_NUM_LIMBS, U16_BITS,
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

pub struct LoadByteAdapterAirInterface;

impl<T> VmAdapterInterface<T> for LoadByteAdapterAirInterface {
    type Reads = [T; BLOCK_FE_WIDTH];
    type Writes = [T; BLOCK_FE_WIDTH];
    type ProcessedInstruction = LoadByteInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadByteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    /// Low 32 bits of the rs1 register, packed as two u16 cells.
    pub rs1_data: [T; PTR_U16_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    /// Destination register pointer.
    pub rd_ptr: T,
    pub read_data_aux: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    /// Low limb of the effective pointer for constraining rs1 + sign_extend(imm).
    pub mem_ptr_low_limb: T,
    /// Carry bit (the parity of the derived high byte-pointer limb) used to convert the aligned
    /// heap *byte* pointer into AS-native u16 *cell* pointer limbs. See
    /// `eval_byte_ptr_limbs_to_u16_cell_ptr_limbs`.
    pub mem_ptr_carry: T,
    pub write_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
    /// Only writes to rd if the load is valid and rd is not x0.
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(LoadByteAdapterCols<u8>)]
pub struct LoadByteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for LoadByteAdapterAir {
    fn width(&self) -> usize {
        LoadByteAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for LoadByteAdapterAir {
    type Interface = LoadByteAdapterAirInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &LoadByteAdapterCols<AB::Var> = local.borrow();

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
        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_block(&local_cols.rs1_data);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local_cols.rs1_ptr),
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

        // Alignment: the aligned heap byte pointer's low limb is divisible by 8, i.e.
        // `aligned_limb / 8 < 2^13`, which also implies `aligned_limb < 2^16`. (The derived high
        // byte limb `mem_ptr_hi` is bounded by the cell-pointer range check below.)
        let block_bytes = AB::F::from_u32(MEMORY_BLOCK_BYTES as u32);
        let aligned_limb = local_cols.mem_ptr_low_limb - shift_amount;
        self.range_bus
            .range_check(
                // aligned_limb / 8 < 2^13 => aligned_limb < 2^16
                aligned_limb.clone() * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());

        // Convert the aligned heap *byte* pointer `[aligned_limb, mem_ptr_hi]` to AS-native u16
        // *cell* pointer limbs (cell = byte / 2) without composing the 32-bit byte pointer into
        // one field element. This inlines `eval_byte_ptr_limbs_to_u16_cell_ptr_limbs` with an
        // unconditional carry bool check, since `is_valid` here is a degree-2 selector expression.
        // The boolean carry plus the `cell_hi` range check force `mem_ptr_carry = mem_ptr_hi & 1`
        // and bound `mem_ptr_hi < 2^(pointer_max_bits - 16)`, i.e. the byte pointer is below
        // `2^pointer_max_bits`.
        builder.assert_bool(local_cols.mem_ptr_carry);
        let inv2 = AB::F::TWO.inverse();
        let mem_ptr_cell_limbs = [
            (aligned_limb + local_cols.mem_ptr_carry * AB::F::from_u32(1u32 << U16_BITS)) * inv2,
            (mem_ptr_hi - local_cols.mem_ptr_carry) * inv2,
        ];
        self.range_bus
            .range_check(
                mem_ptr_cell_limbs[1].clone(),
                cell_ptr_hi_bits(self.pointer_max_bits),
            )
            .eval(builder, is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_u32(MEMORY_AS), mem_ptr_cell_limbs),
                ctx.reads,
                timestamp_pp(),
                &local_cols.read_data_aux,
            )
            .eval(builder, is_valid.clone());

        // Write the loaded value into rd, unless rd is x0.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local_cols.rd_ptr),
                ),
                ctx.writes.clone(),
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, write_count);

        let to_pc = ctx.to_pc.unwrap_or(local_cols.from_state.pc + AB::F::ONE);
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(MEMORY_AS),
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
        let local_cols: &LoadByteAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct LoadByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl LoadByteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut LoadByteAdapterCols<F>,
        compute: impl FnOnce([u16; BLOCK_FE_WIDTH], usize) -> [u16; BLOCK_FE_WIDTH],
    ) -> Result<([u16; BLOCK_FE_WIDTH], usize, [u16; BLOCK_FE_WIDTH]), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != MEMORY_AS
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
        if rs1_ptr > u32::from(u8::MAX) || !rs1_ptr.is_multiple_of(REGISTER_NUM_LIMBS as u32) {
            return Err(PostflightError::new(
                "byte-load instruction has an invalid source register pointer",
            ));
        }
        if rd_ptr > u32::from(u8::MAX) || !rd_ptr.is_multiple_of(REGISTER_NUM_LIMBS as u32) {
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
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(REGISTER_AS, rs1_u16_ptr)?;
        if rs1.value[PTR_U16_LIMBS..].iter().any(|&x| x != 0) {
            return Err(PostflightError::new(
                "byte-load base register exceeds the implemented pointer width",
            ));
        }
        let rs1_val = u32::from(rs1.value[0]) | (u32::from(rs1.value[1]) << U16_BITS);
        let effective_ptr = u32::try_from(address_add_imm(
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
        let memory = replay.read_u16(MEMORY_AS, checked_byte_ptr_to_u16_ptr_value(aligned_ptr)?)?;
        let output = compute(memory.value, shift_amount);

        adapter_row.needs_write = F::from_bool(needs_write);
        if needs_write {
            let write = replay.write_u16(REGISTER_AS, rd_u16_ptr, output)?;
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
        let aligned_byte_limbs = ptr_to_u16_limbs(aligned_ptr).map(u32::from);
        // Alignment check on the aligned low byte limb: `aligned_limb / 8 < 2^13`.
        self.range_checker_chip
            .add_count(aligned_byte_limbs[0] >> 3, U16_BITS - 3);
        // Byte -> cell pointer conversion for the heap block; the AIR range-checks `cell_hi`
        // with `enabled = is_valid`.
        let (mem_carry, cell_limbs) = byte_ptr_limbs_to_cell_ptr_limbs_value(aligned_byte_limbs);
        adapter_row.mem_ptr_carry = F::from_u32(mem_carry);
        self.range_checker_chip
            .add_count(cell_limbs[1], cell_ptr_hi_bits(self.pointer_max_bits));
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
        adapter_row.from_state.pc = F::from_u32(pc_to_idx(from_pc));

        Ok((memory.value, shift_amount, output))
    }
}
