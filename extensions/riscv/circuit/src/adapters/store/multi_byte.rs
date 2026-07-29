use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, Postflight, PostflightError,
        PostflightStep, VmAdapterAir, VmAdapterInterface, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxInput,
        },
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
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    PUBLIC_VALUES_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{
    byte_ptr_to_u16_ptr, checked_byte_ptr_to_u16_ptr_value, expand_to_rv64_block,
    is_multi_byte_access_width, ptr_to_field_u16_limbs, ptr_to_u16_limbs, rv64_address_add_imm,
    sign_extend_imm16, RV64_PTR_U16_LIMBS, RV64_REGISTER_NUM_LIMBS, U16_BITS,
};

pub struct StoreInstruction<T> {
    /// Boolean flag constrained by the core indicating whether this row is active.
    pub is_valid: T,
    /// Absolute opcode number.
    pub opcode: T,
    /// Byte offset of the effective pointer inside the 8-byte memory block.
    pub shift_amount: T,
    /// Boolean flag constrained by the core indicating whether the access spans two blocks.
    pub store_cross: T,
}

pub struct Rv64StoreMultiByteAdapterAirInterface;

/// The previous contents of the two consecutive memory blocks (the second is used only when the
/// access crosses a block boundary), followed by the source register data. The previous contents
/// feed both write auxes, so the core's read-modify-write inputs and the offline checker's
/// receive-side data are the same expressions by construction.
impl<T> VmAdapterInterface<T> for Rv64StoreMultiByteAdapterAirInterface {
    type Reads = ([[T; BLOCK_FE_WIDTH]; 2], [T; BLOCK_FE_WIDTH]);
    type Writes = [[T; BLOCK_FE_WIDTH]; 2];
    type ProcessedInstruction = StoreInstruction<T>;
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
    /// Carry into the high pointer limb for the second block address.
    pub mem_ptr_carry: T,
    /// Timestamp auxiliary columns for the first and optional second block writes. Previous data
    /// is provided by the core chip.
    pub write_base_aux: [MemoryBaseAuxCols<T>; 2],
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
    type Interface = Rv64StoreMultiByteAdapterAirInterface;

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

        // Constrain mem_ptr = rs1 + sign_extend(imm) as a 32-bit addition.
        let inv = AB::F::from_u32(1u32 << U16_BITS).inverse();
        let low_carry =
            (local_cols.rs1_data[0] + local_cols.imm - local_cols.mem_ptr_low_limb) * inv;
        builder.assert_bool(low_carry.clone());
        builder.assert_bool(local_cols.imm_sign);
        let mem_ptr_hi = local_cols.rs1_data[1] + low_carry - local_cols.imm_sign;

        // Prevent mem_ptr overflow while allowing the adapter to write the containing 8-byte block.
        let block_bytes = AB::F::from_u32(MEMORY_BLOCK_BYTES as u32);
        let aligned_limb = local_cols.mem_ptr_low_limb - shift_amount.clone();
        self.range_bus
            .range_check(
                // aligned_limb / 8 < 2^13 => aligned_limb < 2^16
                aligned_limb.clone() * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(mem_ptr_hi.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        // Range check the second block address when the access crosses a block boundary.
        builder.assert_bool(local_cols.mem_ptr_carry);
        let block1_aligned_limb = aligned_limb + block_bytes
            - local_cols.mem_ptr_carry * AB::F::from_u32(1u32 << U16_BITS);
        self.range_bus
            .range_check(block1_aligned_limb * block_bytes.inverse(), U16_BITS - 3)
            .eval(builder, cross.clone());
        // The high limb only needs another range check when the carry increments it.
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
                MemoryWriteAuxInput::from_prev_data_exprs(
                    &local_cols.write_base_aux[0],
                    prev_data0,
                ),
            )
            .eval(builder, is_valid.clone());

        // Write the second block when the access crosses into it. The timestamp slot is consumed
        // either way so the instruction has a static timestamp layout.
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    local_cols.mem_as,
                    byte_ptr_to_u16_ptr::<AB>(
                        mem_ptr - shift_amount + AB::F::from_u32(MEMORY_BLOCK_BYTES as u32),
                    ),
                ),
                write_data1,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(
                    &local_cols.write_base_aux[1],
                    prev_data1,
                ),
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

#[derive(derive_new::new)]
pub struct Rv64StoreMultiByteAdapterFiller {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

type StoreMultiReplay = ([u16; BLOCK_FE_WIDTH], [[u16; BLOCK_FE_WIDTH]; 2], usize);

impl Rv64StoreMultiByteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32, const STORE_WIDTH: usize>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64StoreMultiByteAdapterCols<F>,
        compute: impl FnOnce(
            [u16; BLOCK_FE_WIDTH],
            [[u16; BLOCK_FE_WIDTH]; 2],
            usize,
        ) -> [[u16; BLOCK_FE_WIDTH]; 2],
    ) -> Result<StoreMultiReplay, PostflightError> {
        if !is_multi_byte_access_width(STORE_WIDTH) {
            return Err(PostflightError::new(
                "multi-byte store has an unsupported access width",
            ));
        }
        let instruction = postflight.instruction(step);
        let mem_as = instruction.e.as_canonical_u32();
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || !matches!(mem_as, RV64_MEMORY_AS | PUBLIC_VALUES_AS)
        {
            return Err(PostflightError::new(
                "multi-byte store has invalid address spaces",
            ));
        }
        if !instruction.f.is_one() {
            return Err(PostflightError::new(
                "multi-byte store instruction must be enabled",
            ));
        }
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "multi-byte store has a non-boolean immediate sign",
                ));
            }
        };
        let imm = instruction.c.as_canonical_u32();
        if imm > u16::MAX as u32 {
            return Err(PostflightError::new(
                "multi-byte store immediate exceeds the u16 execution-bus operand",
            ));
        }

        let rs1_ptr = checked_register_pointer(instruction.b.as_canonical_u32(), "rs1")?;
        let rs2_ptr = checked_register_pointer(instruction.a.as_canonical_u32(), "rs2")?;
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(
            RV64_REGISTER_AS,
            checked_byte_ptr_to_u16_ptr_value(u32::from(rs1_ptr))?,
        )?;
        if rs1.value[RV64_PTR_U16_LIMBS..]
            .iter()
            .any(|&limb| limb != 0)
        {
            return Err(PostflightError::new(
                "multi-byte store base register is not a low-32-bit pointer",
            ));
        }
        let rs1_val = u32::from(rs1.value[0]) | (u32::from(rs1.value[1]) << U16_BITS);
        let effective_ptr =
            rv64_address_add_imm(rs1_val, sign_extend_imm16(imm, u32::from(imm_sign)));
        let pointer_limit = 1u64
            .checked_shl(self.pointer_max_bits as u32)
            .unwrap_or(u64::MAX);
        if effective_ptr >= pointer_limit || effective_ptr > u64::from(u32::MAX) {
            return Err(PostflightError::new(
                "multi-byte store effective address exceeds the pointer domain",
            ));
        }
        let effective_ptr = effective_ptr as u32;
        let shift = effective_ptr as usize & (MEMORY_BLOCK_BYTES - 1);
        let aligned_ptr = effective_ptr - shift as u32;
        let crosses = shift + STORE_WIDTH > MEMORY_BLOCK_BYTES;
        if crosses && u64::from(aligned_ptr) + 2 * MEMORY_BLOCK_BYTES as u64 > pointer_limit {
            return Err(PostflightError::new(
                "crossing multi-byte store exceeds the pointer domain",
            ));
        }

        let read_data = replay.read_u16(
            RV64_REGISTER_AS,
            checked_byte_ptr_to_u16_ptr_value(u32::from(rs2_ptr))?,
        )?;
        let block0_ptr = checked_byte_ptr_to_u16_ptr_value(aligned_ptr)?;
        let prev_data0 = replay.peek_u16(mem_as, block0_ptr)?;
        let block1 = if crosses {
            let pointer = checked_byte_ptr_to_u16_ptr_value(
                aligned_ptr
                    .checked_add(MEMORY_BLOCK_BYTES as u32)
                    .ok_or_else(|| {
                        PostflightError::new("crossing multi-byte store pointer overflow")
                    })?,
            )?;
            Some((pointer, replay.peek_u16(mem_as, pointer)?))
        } else {
            None
        };
        let prev_data1 = block1.map_or([0; BLOCK_FE_WIDTH], |(_, value)| value);
        let prev_data = [prev_data0, prev_data1];
        let write_data = compute(read_data.value, prev_data, shift);
        let write0 = replay.write_u16(mem_as, block0_ptr, write_data[0])?;
        if write0.previous_value != prev_data0 {
            return Err(PostflightError::new(
                "multi-byte store first peek did not resolve the write predecessor",
            ));
        }
        let write1 = if crosses {
            let Some((block1_ptr, _)) = block1 else {
                return Err(PostflightError::new(
                    "crossing multi-byte store is missing its second block",
                ));
            };
            let write = replay.write_u16(mem_as, block1_ptr, write_data[1])?;
            if write.previous_value != prev_data1 {
                return Err(PostflightError::new(
                    "multi-byte store second peek did not resolve the write predecessor",
                ));
            }
            Some(write)
        } else {
            replay.advance_timestamp(1)?;
            None
        };
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        if let Some(write1) = write1 {
            mem_helper.fill(
                write1.previous_timestamp,
                write1.timestamp,
                &mut adapter_row.write_base_aux[1],
            );
        } else {
            mem_helper.fill_zero(&mut adapter_row.write_base_aux[1]);
        }
        mem_helper.fill(
            write0.previous_timestamp,
            write0.timestamp,
            &mut adapter_row.write_base_aux[0],
        );

        adapter_row.mem_as = F::from_u32(mem_as);
        let ptr_limbs = ptr_to_u16_limbs(effective_ptr).map(u32::from);
        let aligned_limb = ptr_limbs[0] - shift as u32;
        self.range_checker_chip
            .add_count(aligned_limb >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        adapter_row.mem_ptr_low_limb = F::from_u32(ptr_limbs[0]);
        let block1_low_sum = aligned_limb + MEMORY_BLOCK_BYTES as u32;
        let carry = crosses && block1_low_sum == 1 << U16_BITS;
        adapter_row.mem_ptr_carry = F::from_bool(carry);
        if crosses {
            self.range_checker_chip.add_count(
                (block1_low_sum - (u32::from(carry) << U16_BITS)) >> 3,
                U16_BITS - 3,
            );
        }
        if carry {
            self.range_checker_chip.add_count(
                ptr_limbs[1] + u32::from(carry),
                self.pointer_max_bits - U16_BITS,
            );
        }

        adapter_row.imm_sign = F::from_bool(imm_sign);
        adapter_row.imm = F::from_u32(imm);
        adapter_row.rs2_ptr = F::from_u8(rs2_ptr);
        mem_helper.fill(
            read_data.previous_timestamp,
            read_data.timestamp,
            adapter_row.read_data_aux.as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );
        adapter_row.rs1_data = ptr_to_field_u16_limbs(rs1_val);
        adapter_row.rs1_ptr = F::from_u8(rs1_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok((read_data.value, prev_data, shift))
    }
}

fn checked_register_pointer(pointer: u32, operand: &str) -> Result<u8, PostflightError> {
    if pointer > u8::MAX as u32 || !pointer.is_multiple_of(RV64_REGISTER_NUM_LIMBS as u32) {
        return Err(PostflightError::new(format!(
            "multi-byte store {operand} pointer is not an aligned register address"
        )));
    }
    Ok(pointer as u8)
}
