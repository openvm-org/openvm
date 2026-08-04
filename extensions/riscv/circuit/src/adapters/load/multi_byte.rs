use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, Postflight, PostflightError,
        PostflightStep, VmAdapterAir, VmAdapterInterface, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
        U16_CELL_SIZE,
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
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{
    add_const_u16_limbs_value, address_add_imm, byte_ptr_limbs_to_cell_ptr_limbs_value,
    cell_ptr_hi_bits, checked_byte_ptr_to_u16_ptr_value, checked_register_pointer, expand_to_block,
    is_multi_byte_access_width, ptr_to_field_u16_limbs, ptr_to_u16_limbs,
    reg_byte_ptr_to_cell_ptr_limbs, sign_extend_imm16, PTR_U16_LIMBS, U16_BITS,
};

pub struct LoadInstruction<T> {
    /// Boolean flag constrained by the core indicating whether this row is active.
    pub is_valid: T,
    /// Absolute opcode number.
    pub opcode: T,
    /// Byte offset of the effective pointer inside the 8-byte memory block.
    pub shift_amount: T,
    /// Boolean flag constrained by the core indicating whether the access spans two blocks.
    pub load_cross: T,
}

pub struct LoadMultiByteAdapterAirInterface;

impl<T> VmAdapterInterface<T> for LoadMultiByteAdapterAirInterface {
    /// The memory block containing the effective address, followed by the second block, which is
    /// read only when the access crosses a block boundary.
    type Reads = [[T; BLOCK_FE_WIDTH]; 2];
    type Writes = [[T; BLOCK_FE_WIDTH]; 1];
    type ProcessedInstruction = LoadInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadMultiByteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    /// Low 32 bits of the rs1 register, packed as two u16 cells.
    pub rs1_data: [T; PTR_U16_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    /// Destination register pointer.
    pub rd_ptr: T,
    /// Auxiliary columns for the first and optional second block reads.
    pub read_data_aux: [MemoryReadAuxCols<T>; 2],
    pub imm: T,
    pub imm_sign: T,
    /// Low limb of the effective pointer for constraining rs1 + sign_extend(imm).
    pub mem_ptr_low_limb: T,
    /// Carry bit (the parity of the derived high byte-pointer limb) used to convert the aligned
    /// heap *byte* pointer into AS-native u16 *cell* pointer limbs. See
    /// `eval_byte_ptr_limbs_to_u16_cell_ptr_limbs`.
    pub mem_ptr_carry: T,
    /// Carry into the high cell limb when adding the block stride (in u16 cells) to the first
    /// block's cell pointer to address the second block.
    pub block1_add_carry: T,
    pub write_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
    /// Only writes to rd if the load is valid and rd is not x0.
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(LoadMultiByteAdapterCols<u8>)]
pub struct LoadMultiByteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for LoadMultiByteAdapterAir {
    fn width(&self) -> usize {
        LoadMultiByteAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for LoadMultiByteAdapterAir {
    type Interface = LoadMultiByteAdapterAirInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &LoadMultiByteAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;
        let shift_amount = ctx.instruction.shift_amount;
        let cross = ctx.instruction.load_cross;
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

        // The second block's cell pointer is the first block's plus the block stride (in u16
        // cells), with a boolean carry into the high cell limb (inlined
        // `eval_add_const_u16_limbs`). Range-checking the new low limb to 16 bits forces the
        // carry to be correct; it is only checked when the access crosses a block boundary.
        builder.assert_bool(local_cols.block1_add_carry);
        let cell_stride = AB::F::from_u32((MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32);
        let block1_cell_limbs = [
            mem_ptr_cell_limbs[0].clone() + cell_stride
                - local_cols.block1_add_carry * AB::F::from_u32(1u32 << U16_BITS),
            mem_ptr_cell_limbs[1].clone() + local_cols.block1_add_carry,
        ];
        self.range_bus
            .range_check(block1_cell_limbs[0].clone(), U16_BITS)
            .eval(builder, cross.clone());

        let [read_data0, read_data1] = ctx.reads;
        // Read the memory block containing the effective load address.
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_u32(MEMORY_AS), mem_ptr_cell_limbs),
                read_data0,
                timestamp_pp(),
                &local_cols.read_data_aux[0],
            )
            .eval(builder, is_valid.clone());

        // Read the second block when the access crosses into it. The timestamp slot is consumed
        // either way so the instruction has a static timestamp layout.
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_u32(MEMORY_AS), block1_cell_limbs),
                read_data1,
                timestamp_pp(),
                &local_cols.read_data_aux[1],
            )
            .eval(builder, cross);

        // Write the loaded value into rd, unless rd is x0.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local_cols.rd_ptr),
                ),
                ctx.writes[0].clone(),
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
        let local_cols: &LoadMultiByteAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct LoadMultiByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

type LoadMultiReplay = ([[u16; BLOCK_FE_WIDTH]; 2], usize, [u16; BLOCK_FE_WIDTH]);

impl LoadMultiByteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32, const LOAD_WIDTH: usize>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut LoadMultiByteAdapterCols<F>,
        compute: impl FnOnce([[u16; BLOCK_FE_WIDTH]; 2], usize) -> [u16; BLOCK_FE_WIDTH],
    ) -> Result<LoadMultiReplay, PostflightError> {
        if !is_multi_byte_access_width(LOAD_WIDTH) {
            return Err(PostflightError::new(
                "multi-byte load has an unsupported access width",
            ));
        }
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != MEMORY_AS
        {
            return Err(PostflightError::new(
                "multi-byte load has invalid address spaces",
            ));
        }
        let needs_write = match instruction.f.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "multi-byte load has a non-boolean write enable",
                ));
            }
        };
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "multi-byte load has a non-boolean immediate sign",
                ));
            }
        };
        let imm = instruction.c.as_canonical_u32();
        if imm > u16::MAX as u32 {
            return Err(PostflightError::new(
                "multi-byte load has a non-canonical immediate",
            ));
        }

        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        checked_register_pointer(rs1_ptr)?;
        checked_register_pointer(rd_ptr)?;
        if needs_write != (rd_ptr != 0) {
            return Err(PostflightError::new(
                "multi-byte load write enable does not match its destination",
            ));
        }

        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(REGISTER_AS, checked_byte_ptr_to_u16_ptr_value(rs1_ptr)?)?;
        if rs1.value[PTR_U16_LIMBS..].iter().any(|&cell| cell != 0) {
            return Err(PostflightError::new(
                "multi-byte load base register exceeds the pointer domain",
            ));
        }
        let rs1_val = u32::from(rs1.value[0]) | (u32::from(rs1.value[1]) << U16_BITS);
        let effective_ptr = address_add_imm(rs1_val, sign_extend_imm16(imm, u32::from(imm_sign)));
        let pointer_limit = 1u64
            .checked_shl(self.pointer_max_bits as u32)
            .unwrap_or(u64::MAX);
        if effective_ptr >= pointer_limit || effective_ptr > u64::from(u32::MAX) {
            return Err(PostflightError::new(
                "multi-byte load effective address exceeds the pointer domain",
            ));
        }
        let effective_ptr = effective_ptr as u32;
        let shift = (effective_ptr as usize) & (MEMORY_BLOCK_BYTES - 1);
        let aligned_ptr = effective_ptr - shift as u32;
        let crosses = shift + LOAD_WIDTH > MEMORY_BLOCK_BYTES;
        if crosses && u64::from(aligned_ptr) + 2 * MEMORY_BLOCK_BYTES as u64 > pointer_limit {
            return Err(PostflightError::new(
                "crossing multi-byte load exceeds the pointer domain",
            ));
        }

        let block0 = replay.read_u16(MEMORY_AS, checked_byte_ptr_to_u16_ptr_value(aligned_ptr)?)?;
        let block1 = if crosses {
            Some(replay.read_u16(
                MEMORY_AS,
                checked_byte_ptr_to_u16_ptr_value(aligned_ptr + MEMORY_BLOCK_BYTES as u32)?,
            )?)
        } else {
            replay.advance_timestamp(1)?;
            None
        };
        let read_data = [
            block0.value,
            block1.as_ref().map_or([0; BLOCK_FE_WIDTH], |x| x.value),
        ];
        let output = compute(read_data, shift);
        let write = if needs_write {
            Some(replay.write_u16(
                REGISTER_AS,
                checked_byte_ptr_to_u16_ptr_value(rd_ptr)?,
                output,
            )?)
        } else {
            replay.advance_timestamp(1)?;
            None
        };
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        adapter_row.needs_write = F::from_bool(needs_write);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        if let Some(write) = write {
            adapter_row
                .write_aux
                .set_prev_data(write.previous_value.map(F::from_u16));
            mem_helper.fill(
                write.previous_timestamp,
                write.timestamp,
                adapter_row.write_aux.as_mut(),
            );
        } else {
            adapter_row.rd_ptr = F::ZERO;
            adapter_row.write_aux.prev_data = [F::ZERO; BLOCK_FE_WIDTH];
            mem_helper.fill_zero(&mut adapter_row.write_aux.base);
        }

        let ptr_limbs = ptr_to_u16_limbs(effective_ptr).map(u32::from);
        let aligned_limb = ptr_limbs[0] - shift as u32;
        // Alignment check on the aligned low byte limb: `aligned_limb / 8 < 2^13`.
        self.range_checker_chip
            .add_count(aligned_limb >> 3, U16_BITS - 3);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);
        // Byte -> cell pointer conversion for the first block; the AIR range-checks `cell_hi`
        // with `enabled = is_valid`.
        let (mem_carry, cell_limbs) =
            byte_ptr_limbs_to_cell_ptr_limbs_value([aligned_limb, ptr_limbs[1]]);
        adapter_row.mem_ptr_carry = F::from_u32(mem_carry);
        self.range_checker_chip
            .add_count(cell_limbs[1], cell_ptr_hi_bits(self.pointer_max_bits));
        // Second-block cell pointer carry and low-limb range check (AIR `enabled = cross`).
        if crosses {
            let (add_carry, block1_cell_limbs) =
                add_const_u16_limbs_value(cell_limbs, (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32);
            adapter_row.block1_add_carry = F::from_u32(add_carry);
            self.range_checker_chip
                .add_count(block1_cell_limbs[0], U16_BITS);
        } else {
            adapter_row.block1_add_carry = F::ZERO;
        }

        adapter_row.imm = F::from_u32(imm);
        adapter_row.imm_sign = F::from_bool(imm_sign);
        if let Some(block1) = block1 {
            mem_helper.fill(
                block1.previous_timestamp,
                block1.timestamp,
                adapter_row.read_data_aux[1].as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.read_data_aux[1].as_mut());
        }
        mem_helper.fill(
            block0.previous_timestamp,
            block0.timestamp,
            adapter_row.read_data_aux[0].as_mut(),
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

        Ok((read_data, shift, output))
    }
}
