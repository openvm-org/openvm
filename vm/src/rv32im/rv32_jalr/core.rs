use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::{XorBus, XorLookupChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32JalrOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::{
        compose, JumpUiProcessedInstruction, PC_BITS, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    system::program::Instruction,
};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalrCoreCols<T> {
    pub imm: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    // To save a column, we only store the 3 most significant limbs of `rd_data`
    // the least significant limb can be derived using from_pc and the other limbs
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub is_valid: T,
    // Used to range check that rd_data elements are bytes with XorBus
    pub xor_res: T,

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2`.
    pub to_pc_limbs: [T; 2],
    pub imm_sign: T,
}

pub struct Rv32JalrCoreRecord<F> {
    pub imm: F,
    pub rs1_data: [F; RV32_REGISTER_NUM_LIMBS],
    pub rd_data: [F; RV32_REGISTER_NUM_LIMBS - 1],
    pub xor_res: F,
    pub to_pc_least_sig_bit: F,
    pub to_pc_limbs: [F; 2],
    pub imm_sign: F,
}

#[derive(Debug, Clone)]
pub struct Rv32JalrCoreAir {
    // XorBus is used to range check that rd_data elements are bytes
    pub xor_bus: XorBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32JalrCoreAir {
    fn width(&self) -> usize {
        Rv32JalrCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JalrCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JalrCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<JumpUiProcessedInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32JalrCoreCols<AB::Var> = (*local_core).borrow();
        let Rv32JalrCoreCols::<AB::Var> {
            imm,
            rs1_data: rs1,
            rd_data: rd,
            is_valid,
            xor_res,
            imm_sign,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // composed = composed(rd_data) - rd[0]
        let composed = rd
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });

        let least_sig_limb = from_pc + AB::F::from_canonical_u32(4) - composed;

        // rd_data is the final data needed
        let rd_data = array::from_fn(|i| {
            if i == 0 {
                least_sig_limb.clone()
            } else {
                rd[i - 1].into().clone()
            }
        });

        // Constrain rd_data
        // Assumes only from_pc in [0,2^PC_BITS) is allowed by program bus
        // Xor bus is being used to range check two bytes simultaneously. The xor result itself is not used.
        self.xor_bus
            .send(rd_data[0].clone(), rd_data[1].clone(), xor_res)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[2].clone(), RV32_CELL_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[3].clone(), PC_BITS - RV32_CELL_BITS * 3)
            .eval(builder, is_valid);

        builder.assert_bool(imm_sign);

        // constrain to_pc_least_sig_bit + 2 * to_pc_limbs = rs1 + imm as a i32 addition with 2 limbs
        // RISC-V spec explicitly sets the least significant bit of `to_pc` to 0
        let rs1_limbs_01 = rs1[0] + rs1[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let rs1_limbs_23 = rs1[2] + rs1[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let inv = AB::F::from_canonical_u32(1 << 16).inverse();

        builder.assert_bool(to_pc_least_sig_bit);
        let carry =
            (rs1_limbs_01 + imm - to_pc_limbs[0] * AB::F::two() - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        let imm_extend_limb = imm_sign * AB::F::from_canonical_u32((1 << 16) - 1);
        let carry = (rs1_limbs_23 + imm_extend_limb + carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry);

        // preventing to_pc overflow
        self.range_bus
            .range_check(to_pc_limbs[1], PC_BITS - 16)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0], 15)
            .eval(builder, is_valid);
        let to_pc =
            to_pc_limbs[0] * AB::F::two() + to_pc_limbs[1] * AB::F::from_canonical_u32(1 << 16);

        let expected_opcode = AB::F::from_canonical_usize(JALR as usize + self.offset);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [rs1.map(|x| x.into())].into(),
            writes: [rd_data].into(),
            instruction: JumpUiProcessedInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode.into(),
                immediate: imm.into(),
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32JalrCoreChip {
    pub air: Rv32JalrCoreAir,
    pub xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl Rv32JalrCoreChip {
    pub fn new(
        xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        assert!(range_checker_chip.bus().range_max_bits >= 15);
        Self {
            air: Rv32JalrCoreAir {
                xor_bus: xor_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
                offset,
            },
            xor_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32JalrCoreChip
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = Rv32JalrCoreRecord<F>;
    type Air = Rv32JalrCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        assert!(self.range_checker_chip.range_max_bits() >= 16);
        let Instruction { opcode, c, .. } = *instruction;
        let local_opcode_index = Rv32JalrOpcode::from_usize(opcode - self.air.offset);

        let imm = c.as_canonical_u32();
        let imm_sign = (imm & 0x8000) >> 15;
        let imm_extended = imm + imm_sign * 0xffff0000;

        let rs1 = reads.into()[0];
        let rs1_val = compose(rs1);

        let (to_pc, rd_data) = run_jalr(local_opcode_index, from_pc, imm_extended, rs1_val);

        let xor_res = F::from_canonical_u32(self.xor_lookup_chip.request(rd_data[0], rd_data[1]));
        self.range_checker_chip
            .add_count(rd_data[2], RV32_CELL_BITS);
        self.range_checker_chip
            .add_count(rd_data[3], PC_BITS - RV32_CELL_BITS * 3);

        let mask = (1 << 15) - 1;
        let to_pc_least_sig_bit = rs1_val.wrapping_add(imm_extended) & 1;

        let to_pc_limbs = array::from_fn(|i| F::from_canonical_u32((to_pc >> (1 + i * 15)) & mask));
        self.range_checker_chip
            .add_count(to_pc_limbs[0].as_canonical_u32(), 15);
        self.range_checker_chip
            .add_count(to_pc_limbs[1].as_canonical_u32(), 14);

        let rd_data = rd_data.map(F::from_canonical_u32);

        let output = AdapterRuntimeContext {
            to_pc: Some(to_pc),
            writes: [rd_data].into(),
        };

        Ok((
            output,
            Rv32JalrCoreRecord {
                imm: c,
                rd_data: array::from_fn(|i| rd_data[i + 1]),
                rs1_data: rs1,
                xor_res,
                to_pc_least_sig_bit: F::from_canonical_u32(to_pc_least_sig_bit),
                to_pc_limbs,
                imm_sign: F::from_canonical_u32(imm_sign),
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", Rv32JalrOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32JalrCoreCols<F> = row_slice.borrow_mut();
        core_cols.imm = record.imm;
        core_cols.rd_data = record.rd_data;
        core_cols.rs1_data = record.rs1_data;
        core_cols.xor_res = record.xor_res;
        core_cols.to_pc_least_sig_bit = record.to_pc_least_sig_bit;
        core_cols.to_pc_limbs = record.to_pc_limbs;
        core_cols.imm_sign = record.imm_sign;
        core_cols.is_valid = F::one();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns (to_pc, rd_data)
pub(super) fn run_jalr(
    _opcode: Rv32JalrOpcode,
    pc: u32,
    imm: u32,
    rs1: u32,
) -> (u32, [u32; RV32_REGISTER_NUM_LIMBS]) {
    let to_pc = rs1.wrapping_add(imm);
    let to_pc = to_pc - (to_pc & 1);
    assert!(to_pc < (1 << PC_BITS));
    (
        to_pc,
        array::from_fn(|i: usize| ((pc + 4) >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX),
    )
}
