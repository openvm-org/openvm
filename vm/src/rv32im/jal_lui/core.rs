use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::xor::{XorBus, XorLookupChip};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::{instruction::Instruction, program::PC_BITS};
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32JalLuiOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, ImmInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    rv32im::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, RV_J_TYPE_IMM_BITS},
};
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalLuiCoreCols<T> {
    pub imm: T,
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_jal: T,
    pub is_lui: T,
    pub xor_res: T,
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiCoreAir {
    // XorBus is used to range check that rd_data elements are bytes
    pub bus: XorBus,
    offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32JalLuiCoreAir {
    fn width(&self) -> usize {
        Rv32JalLuiCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JalLuiCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JalLuiCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32JalLuiCoreCols<AB::Var> = (*local_core).borrow();
        let Rv32JalLuiCoreCols::<AB::Var> {
            imm,
            rd_data: rd,
            is_jal,
            is_lui,
            xor_res,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.when(is_lui).assert_zero(rd[0]);

        self.bus
            .send(rd[1], rd[2], xor_res)
            .eval(builder, is_valid.clone());
        self.bus
            .send(rd[0], rd[3] * is_lui, rd[0] + rd[3] * is_lui)
            .eval(builder, is_valid.clone());

        // In case of JAL constrain that last limb has at most [last_limb_bits] bits

        let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1 << x));
        let additional_bits = AB::F::from_canonical_u32(additional_bits);
        self.bus
            .send(rd[3], additional_bits, rd[3] + additional_bits)
            .eval(builder, is_jal);

        let intermed_val = rd
            .iter()
            .skip(1)
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });

        builder.when(is_lui).assert_eq(
            intermed_val.clone(),
            imm * AB::F::from_canonical_u32(1 << (12 - RV32_CELL_BITS)),
        );

        let intermed_val = rd[0] + intermed_val * AB::Expr::from_canonical_u32(1 << RV32_CELL_BITS);
        builder
            .when(is_jal)
            .assert_eq(intermed_val, from_pc + AB::F::from_canonical_u32(4));

        let to_pc = from_pc + is_lui * AB::F::from_canonical_u32(4) + is_jal * imm;

        let expected_opcode = is_lui * AB::F::from_canonical_u32(LUI as u32)
            + is_jal * AB::F::from_canonical_u32(JAL as u32)
            + AB::F::from_canonical_usize(self.offset);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [].into(),
            writes: [rd.map(|x| x.into())].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm.into(),
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiCoreRecord<F: Field> {
    pub rd_data: [F; RV32_REGISTER_NUM_LIMBS],
    pub imm: F,
    pub is_jal: bool,
    pub is_lui: bool,
}

#[derive(Debug, Clone)]
pub struct Rv32JalLuiCoreChip {
    pub air: Rv32JalLuiCoreAir,
    pub xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
}

impl Rv32JalLuiCoreChip {
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>, offset: usize) -> Self {
        Self {
            air: Rv32JalLuiCoreAir {
                bus: xor_lookup_chip.bus(),
                offset,
            },
            xor_lookup_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32JalLuiCoreChip
where
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = Rv32JalLuiCoreRecord<F>;
    type Air = Rv32JalLuiCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode_index = Rv32JalLuiOpcode::from_usize(instruction.opcode - self.air.offset);
        let imm = instruction.c;

        let signed_imm = match local_opcode_index {
            JAL => {
                // Note: signed_imm is a signed integer and imm is a field element
                (imm + F::from_canonical_u32(1 << (RV_J_TYPE_IMM_BITS - 1))).as_canonical_u32()
                    as i32
                    - (1 << (RV_J_TYPE_IMM_BITS - 1))
            }
            LUI => imm.as_canonical_u32() as i32,
        };
        let (to_pc, rd_data) = run_jal_lui(local_opcode_index, from_pc, signed_imm);

        self.xor_lookup_chip.request(rd_data[1], rd_data[2]);
        if local_opcode_index == JAL {
            self.xor_lookup_chip.request(rd_data[0], 0);
            let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
            let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1 << x));
            self.xor_lookup_chip.request(rd_data[3], additional_bits);
        } else if local_opcode_index == LUI {
            self.xor_lookup_chip.request(0, rd_data[3]);
        }

        let rd_data = rd_data.map(F::from_canonical_u32);

        let output = AdapterRuntimeContext {
            to_pc: Some(to_pc),
            writes: [rd_data].into(),
        };

        Ok((
            output,
            Rv32JalLuiCoreRecord {
                rd_data,
                imm,
                is_jal: local_opcode_index == JAL,
                is_lui: local_opcode_index == LUI,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalLuiOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32JalLuiCoreCols<F> = row_slice.borrow_mut();
        core_cols.rd_data = record.rd_data;
        core_cols.imm = record.imm;
        core_cols.is_jal = F::from_bool(record.is_jal);
        core_cols.is_lui = F::from_bool(record.is_lui);
        let x = core_cols.rd_data[1].as_canonical_u32();
        let y = core_cols.rd_data[2].as_canonical_u32();
        core_cols.xor_res = F::from_canonical_u32(x ^ y);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns (to_pc, rd_data)
pub(super) fn run_jal_lui(
    opcode: Rv32JalLuiOpcode,
    pc: u32,
    imm: i32,
) -> (u32, [u32; RV32_REGISTER_NUM_LIMBS]) {
    match opcode {
        JAL => {
            let rd_data = array::from_fn(|i| ((pc + 4) >> (8 * i)) & ((1 << RV32_CELL_BITS) - 1));
            let next_pc = pc as i32 + imm;
            assert!(next_pc >= 0);
            (next_pc as u32, rd_data)
        }
        LUI => {
            let imm = imm as u32;
            let rd = imm << 12;
            let rd_data =
                array::from_fn(|i| (rd >> (RV32_CELL_BITS * i)) & ((1 << RV32_CELL_BITS) - 1));
            (pc + 4, rd_data)
        }
    }
}
