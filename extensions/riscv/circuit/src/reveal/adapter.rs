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
    program::DEFAULT_PC_STEP, riscv::REGISTER_AS, LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::REVEAL_ACCESS_WIDTH;
use crate::adapters::{
    byte_ptr_to_u16_ptr, checked_byte_ptr_to_u16_ptr_value, checked_register_pointer,
    expand_to_block, ptr_to_field_u16_limbs, ptr_to_u16_limbs, address_add_imm,
    bytes_to_u16_block, u16_block_to_bytes, sign_extend_imm16, PTR_U16_LIMBS,
    U16_BITS,
};

pub struct RevealInstruction<T> {
    pub is_valid: T,
    pub opcode: T,
    pub shift_amount: T,
    pub crosses_block: T,
}

pub struct RevealAdapterAirInterface;

impl<T> VmAdapterInterface<T> for RevealAdapterAirInterface {
    type Reads = ([[T; BLOCK_FE_WIDTH]; 2], [T; BLOCK_FE_WIDTH]);
    type Writes = [[T; BLOCK_FE_WIDTH]; 2];
    type ProcessedInstruction = RevealInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RevealAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub base_ptr: T,
    pub base_data: [T; PTR_U16_LIMBS],
    pub base_aux_cols: MemoryReadAuxCols<T>,
    pub src_ptr: T,
    pub src_aux_cols: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    pub reveal_ptr_low_limb: T,
    pub reveal_ptr_carry: T,
    pub write_base_aux: [MemoryBaseAuxCols<T>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(RevealAdapterCols<u8>)]
pub struct RevealAdapterAir {
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

#[derive(derive_new::new, Clone)]
pub struct RevealAdapterFiller {
    pointer_max_bits: usize,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

pub(crate) struct RevealReplay {
    pub src_data: [u16; BLOCK_FE_WIDTH],
    pub prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
    pub shift: usize,
}

fn reveal_write_data(
    src_data: [u16; BLOCK_FE_WIDTH],
    prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [[u16; BLOCK_FE_WIDTH]; 2] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&u16_block_to_bytes(prev_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&u16_block_to_bytes(prev_data[1]));
    let value = u16_block_to_bytes(src_data);
    bytes[byte_shift..byte_shift + REVEAL_ACCESS_WIDTH]
        .copy_from_slice(&value[..REVEAL_ACCESS_WIDTH]);
    [
        bytes_to_u16_block(bytes[..2 * BLOCK_FE_WIDTH].try_into().unwrap()),
        bytes_to_u16_block(bytes[2 * BLOCK_FE_WIDTH..].try_into().unwrap()),
    ]
}

impl<F: Field> BaseAir<F> for RevealAdapterAir {
    fn width(&self) -> usize {
        RevealAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for RevealAdapterAir {
    type Interface = RevealAdapterAirInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &RevealAdapterCols<AB::Var> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;
        let shift_amount = ctx.instruction.shift_amount;
        let crosses_block = ctx.instruction.crosses_block;

        let base_data: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_block(&cols.base_data);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(cols.base_ptr),
                ),
                base_data,
                timestamp_pp(),
                &cols.base_aux_cols,
            )
            .eval(builder, is_valid.clone());

        let inv = AB::F::from_u32(1u32 << U16_BITS).inverse();
        let low_carry = (cols.base_data[0] + cols.imm - cols.reveal_ptr_low_limb) * inv;
        builder.assert_bool(low_carry.clone());
        builder.assert_bool(cols.imm_sign);
        let reveal_ptr_hi = cols.base_data[1] + low_carry - cols.imm_sign;

        let block_bytes = AB::F::from_u32(MEMORY_BLOCK_BYTES as u32);
        let aligned_limb = cols.reveal_ptr_low_limb - shift_amount.clone();
        self.range_bus
            .range_check(aligned_limb.clone() * block_bytes.inverse(), U16_BITS - 3)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(reveal_ptr_hi.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        builder.assert_bool(cols.reveal_ptr_carry);
        let block1_aligned_limb =
            aligned_limb + block_bytes - cols.reveal_ptr_carry * AB::F::from_u32(1u32 << U16_BITS);
        self.range_bus
            .range_check(block1_aligned_limb * block_bytes.inverse(), U16_BITS - 3)
            .eval(builder, crosses_block.clone());
        self.range_bus
            .range_check(
                reveal_ptr_hi.clone() + cols.reveal_ptr_carry,
                self.pointer_max_bits - U16_BITS,
            )
            .eval(builder, cols.reveal_ptr_carry);

        let reveal_ptr =
            cols.reveal_ptr_low_limb + reveal_ptr_hi * AB::F::from_u32(1u32 << U16_BITS);
        let (prev_data, src_data) = ctx.reads;
        let [prev_data0, prev_data1] = prev_data;
        let [write_data0, write_data1] = ctx.writes;

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(cols.src_ptr),
                ),
                src_data,
                timestamp_pp(),
                &cols.src_aux_cols,
            )
            .eval(builder, is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(PUBLIC_VALUES_AS),
                    byte_ptr_to_u16_ptr::<AB>(reveal_ptr.clone() - shift_amount.clone()),
                ),
                write_data0,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(&cols.write_base_aux[0], prev_data0),
            )
            .eval(builder, is_valid.clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(PUBLIC_VALUES_AS),
                    byte_ptr_to_u16_ptr::<AB>(
                        reveal_ptr - shift_amount + AB::F::from_u32(MEMORY_BLOCK_BYTES as u32),
                    ),
                ),
                write_data1,
                timestamp_pp(),
                MemoryWriteAuxInput::from_prev_data_exprs(&cols.write_base_aux[1], prev_data1),
            )
            .eval(builder, crosses_block);

        let to_pc = ctx
            .to_pc
            .unwrap_or(cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    cols.src_ptr.into(),
                    cols.base_ptr.into(),
                    cols.imm.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(PUBLIC_VALUES_AS),
                    is_valid.clone(),
                    cols.imm_sign.into(),
                ],
                cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_usize(timestamp_delta),
                },
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &RevealAdapterCols<AB::Var> = local.borrow();
        cols.from_state.pc
    }
}

impl RevealAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        cols: &mut RevealAdapterCols<F>,
    ) -> Result<RevealReplay, PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.opcode != RevealOpcode::REVEAL.global_opcode()
            || instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != PUBLIC_VALUES_AS
        {
            return Err(PostflightError::new(
                "reveal instruction has invalid address spaces",
            ));
        }
        if !instruction.f.is_one() {
            return Err(PostflightError::new("reveal instruction must be enabled"));
        }
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "reveal instruction has a non-boolean immediate sign",
                ));
            }
        };
        let imm = instruction.c.as_canonical_u32();
        if imm > u16::MAX as u32 {
            return Err(PostflightError::new(
                "reveal immediate exceeds the u16 execution-bus operand",
            ));
        }

        let base_ptr = checked_register_pointer(instruction.b.as_canonical_u32())?;
        let src_ptr = checked_register_pointer(instruction.a.as_canonical_u32())?;
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let mut replay = postflight.replay(step);
        let base = replay.read_u16(
            REGISTER_AS,
            checked_byte_ptr_to_u16_ptr_value(u32::from(base_ptr))?,
        )?;
        if base.value[PTR_U16_LIMBS..]
            .iter()
            .any(|&limb| limb != 0)
        {
            return Err(PostflightError::new(
                "reveal base register is not a low-32-bit pointer",
            ));
        }
        let base_value = u32::from(base.value[0]) | (u32::from(base.value[1]) << U16_BITS);
        let effective_ptr =
            address_add_imm(base_value, sign_extend_imm16(imm, u32::from(imm_sign)));
        let pointer_limit = 1u64
            .checked_shl(self.pointer_max_bits as u32)
            .unwrap_or(u64::MAX);
        if effective_ptr >= pointer_limit || effective_ptr > u64::from(u32::MAX) {
            return Err(PostflightError::new(
                "reveal address exceeds the pointer domain",
            ));
        }
        let effective_ptr = effective_ptr as u32;
        let shift = effective_ptr as usize & (MEMORY_BLOCK_BYTES - 1);
        let aligned_ptr = effective_ptr - shift as u32;
        let crosses_block = shift + REVEAL_ACCESS_WIDTH > MEMORY_BLOCK_BYTES;
        if crosses_block && u64::from(aligned_ptr) + 2 * MEMORY_BLOCK_BYTES as u64 > pointer_limit {
            return Err(PostflightError::new(
                "crossing reveal exceeds the pointer domain",
            ));
        }

        let src_data = replay.read_u16(
            REGISTER_AS,
            checked_byte_ptr_to_u16_ptr_value(u32::from(src_ptr))?,
        )?;
        let block0_ptr = checked_byte_ptr_to_u16_ptr_value(aligned_ptr)?;
        let prev_data0 = replay.peek_u16(PUBLIC_VALUES_AS, block0_ptr)?;
        let block1 = if crosses_block {
            let pointer = checked_byte_ptr_to_u16_ptr_value(
                aligned_ptr
                    .checked_add(MEMORY_BLOCK_BYTES as u32)
                    .ok_or_else(|| PostflightError::new("crossing reveal pointer overflow"))?,
            )?;
            Some((pointer, replay.peek_u16(PUBLIC_VALUES_AS, pointer)?))
        } else {
            None
        };
        let prev_data1 = block1.map_or([0; BLOCK_FE_WIDTH], |(_, value)| value);
        let prev_data = [prev_data0, prev_data1];
        let write_data = reveal_write_data(src_data.value, prev_data, shift);
        let write0 = replay.write_u16(PUBLIC_VALUES_AS, block0_ptr, write_data[0])?;
        if write0.previous_value != prev_data0 {
            return Err(PostflightError::new(
                "reveal first peek did not resolve the write predecessor",
            ));
        }
        let write1 = if let Some((block1_ptr, _)) = block1 {
            let write = replay.write_u16(PUBLIC_VALUES_AS, block1_ptr, write_data[1])?;
            if write.previous_value != prev_data1 {
                return Err(PostflightError::new(
                    "reveal second peek did not resolve the write predecessor",
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
                &mut cols.write_base_aux[1],
            );
        } else {
            mem_helper.fill_zero(&mut cols.write_base_aux[1]);
        }
        mem_helper.fill(
            write0.previous_timestamp,
            write0.timestamp,
            &mut cols.write_base_aux[0],
        );

        let ptr_limbs = ptr_to_u16_limbs(effective_ptr).map(u32::from);
        let aligned_limb = ptr_limbs[0] - shift as u32;
        self.range_checker_chip
            .add_count(aligned_limb >> 3, U16_BITS - 3);
        self.range_checker_chip
            .add_count(ptr_limbs[1], self.pointer_max_bits - U16_BITS);
        cols.reveal_ptr_low_limb = F::from_u32(ptr_limbs[0]);
        let block1_low_sum = aligned_limb + MEMORY_BLOCK_BYTES as u32;
        let carry = crosses_block && block1_low_sum == 1 << U16_BITS;
        cols.reveal_ptr_carry = F::from_bool(carry);
        if crosses_block {
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

        cols.imm_sign = F::from_bool(imm_sign);
        cols.imm = F::from_u32(imm);
        cols.src_ptr = F::from_u8(src_ptr);
        mem_helper.fill(
            src_data.previous_timestamp,
            src_data.timestamp,
            cols.src_aux_cols.as_mut(),
        );
        mem_helper.fill(
            base.previous_timestamp,
            base.timestamp,
            cols.base_aux_cols.as_mut(),
        );
        cols.base_data = ptr_to_field_u16_limbs(base_value);
        cols.base_ptr = F::from_u8(base_ptr);
        cols.from_state.timestamp = F::from_u32(from_timestamp);
        cols.from_state.pc = F::from_u32(from_pc);

        Ok(RevealReplay {
            src_data: src_data.value,
            prev_data,
            shift,
        })
    }
}
