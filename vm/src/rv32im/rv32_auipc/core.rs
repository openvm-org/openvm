use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32AuipcOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::{JumpUiProcessedInstruction, RV32_CELL_BITS, RV32_REGISTER_NUM_LANES},
    system::program::Instruction,
};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32AuipcCols<T> {
    pub is_valid: T,
    pub imm_limbs: [T; RV32_REGISTER_NUM_LANES - 1],
    pub pc_limbs: [T; RV32_REGISTER_NUM_LANES - 1],
    pub rd_data: [T; RV32_REGISTER_NUM_LANES],

    // Used to constrain rd_data to 8-bits with XorBus
    pub rd_byte_check: [T; RV32_REGISTER_NUM_LANES / 2],

    // Used to constrain pc_limbs and imm_limbs to 8-bits with XorBus
    pub pc_imm_byte_check: [T; RV32_REGISTER_NUM_LANES - 2],
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcCoreAir {
    pub bus: XorBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32AuipcCoreAir {
    fn width(&self) -> usize {
        Rv32AuipcCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32AuipcCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LANES]; 1]>,
    I::ProcessedInstruction: From<JumpUiProcessedInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32AuipcCols<AB::Var> = (*local_core).borrow();

        let Rv32AuipcCols {
            is_valid,
            imm_limbs,
            pc_limbs,
            rd_data,
            rd_byte_check,
            pc_imm_byte_check,
        } = *cols;
        builder.assert_bool(is_valid);
        let intermed_val = pc_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });
        let imm = imm_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });

        builder
            .when(cols.is_valid)
            .assert_eq(rd_data[0], from_pc - intermed_val);

        let mut carry: [AB::Expr; RV32_REGISTER_NUM_LANES] = array::from_fn(|_| AB::Expr::zero());
        let carry_divide = AB::F::from_canonical_usize(1 << RV32_CELL_BITS).inverse();

        for i in 1..RV32_REGISTER_NUM_LANES {
            carry[i] = AB::Expr::from(carry_divide)
                * (pc_limbs[i - 1] + imm_limbs[i - 1] - rd_data[i] + carry[i - 1].clone());
            builder.when(is_valid).assert_bool(carry[i].clone());
        }

        // Range checking to 8 bits
        for i in 0..RV32_REGISTER_NUM_LANES / 2 {
            self.bus
                .send(rd_data[i * 2], rd_data[i * 2 + 1], rd_byte_check[i])
                .eval(builder, is_valid);
        }
        let limbs = [imm_limbs, pc_limbs].concat();
        for i in 0..RV32_REGISTER_NUM_LANES - 2 {
            self.bus
                .send(limbs[i * 2], limbs[i * 2 + 1], pc_imm_byte_check[i])
                .eval(builder, is_valid);
        }

        let expected_opcode = AB::F::from_canonical_usize(AUIPC as usize + self.offset);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [rd_data.map(|x| x.into())].into(),
            instruction: JumpUiProcessedInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode.into(),
                immediate: imm,
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcCoreRecord<F> {
    pub imm_limbs: [F; RV32_REGISTER_NUM_LANES - 1],
    pub pc_limbs: [F; RV32_REGISTER_NUM_LANES - 1],
    pub rd_data: [F; RV32_REGISTER_NUM_LANES],
    pub rd_byte_check: [F; RV32_REGISTER_NUM_LANES / 2],
    pub pc_imm_byte_check: [F; RV32_REGISTER_NUM_LANES - 2],
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcCoreChip {
    pub air: Rv32AuipcCoreAir,
    pub xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>,
}

impl Rv32AuipcCoreChip {
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<RV32_CELL_BITS>>, offset: usize) -> Self {
        Self {
            air: Rv32AuipcCoreAir {
                bus: xor_lookup_chip.bus(),
                offset,
            },
            xor_lookup_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32AuipcCoreChip
where
    I::Writes: From<[[F; RV32_REGISTER_NUM_LANES]; 1]>,
{
    type Record = Rv32AuipcCoreRecord<F>;
    type Air = Rv32AuipcCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode_index = Rv32AuipcOpcode::from_usize(instruction.opcode - self.air.offset);
        let imm = instruction.op_c.as_canonical_u32();
        let rd_data = solve_auipc(local_opcode_index, from_pc, imm);
        let rd_data_field = rd_data.map(F::from_canonical_u32);

        let output = AdapterRuntimeContext::without_pc([rd_data_field]);

        let imm_limbs = array::from_fn(|i| (imm >> (i * RV32_CELL_BITS)) & RV32_LIMB_MAX);
        let pc_limbs = array::from_fn(|i| (from_pc >> ((i + 1) * RV32_CELL_BITS)) & RV32_LIMB_MAX);

        let rd_byte_check = array::from_fn(|i| {
            self.xor_lookup_chip
                .request(rd_data[i * 2], rd_data[i * 2 + 1])
        });

        let limbs: Vec<u32> = [imm_limbs, pc_limbs].concat();
        let pc_imm_byte_check =
            array::from_fn(|i| self.xor_lookup_chip.request(limbs[i * 2], limbs[i * 2 + 1]));

        Ok((
            output,
            Self::Record {
                imm_limbs: imm_limbs.map(F::from_canonical_u32),
                pc_limbs: pc_limbs.map(F::from_canonical_u32),
                rd_data: rd_data.map(F::from_canonical_u32),
                rd_byte_check: rd_byte_check.map(F::from_canonical_u32),
                pc_imm_byte_check: pc_imm_byte_check.map(F::from_canonical_u32),
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32AuipcOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32AuipcCols<F> = row_slice.borrow_mut();
        core_cols.imm_limbs = record.imm_limbs;
        core_cols.pc_limbs = record.pc_limbs;
        core_cols.rd_data = record.rd_data;
        core_cols.rd_byte_check = record.rd_byte_check;
        core_cols.pc_imm_byte_check = record.pc_imm_byte_check;
        core_cols.is_valid = F::one();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns rd_data
pub(super) fn solve_auipc(
    _opcode: Rv32AuipcOpcode,
    pc: u32,
    imm: u32,
) -> [u32; RV32_REGISTER_NUM_LANES] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    array::from_fn(|i| (rd >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX)
}
