use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, Result, SignedImmInstruction,
        SingleTraceStep, StepExecutorE1, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalrOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

use crate::adapters::{compose, Rv32JalrAdapterCols, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

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

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2`.
    pub to_pc_limbs: [T; 2],
    pub imm_sign: T,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct Rv32JalrCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
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
    I::ProcessedInstruction: From<SignedImmInstruction<AB::Expr>>,
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
            imm_sign,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // composed is the composition of 3 most significant limbs of rd
        let composed = rd
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });

        let least_sig_limb = from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP) - composed;

        // rd_data is the final decomposition of `from_pc + DEFAULT_PC_STEP` we need.
        // The range check on `least_sig_limb` also ensures that `rd_data` correctly represents
        // `from_pc + DEFAULT_PC_STEP`. Specifically, if `rd_data` does not match the
        // expected limb, then `least_sig_limb` becomes the real `least_sig_limb` plus the
        // difference between `composed` and the three most significant limbs of `from_pc +
        // DEFAULT_PC_STEP`. In that case, `least_sig_limb` >= 2^RV32_CELL_BITS.
        let rd_data = array::from_fn(|i| {
            if i == 0 {
                least_sig_limb.clone()
            } else {
                rd[i - 1].into().clone()
            }
        });

        // Constrain rd_data
        // Assumes only from_pc in [0,2^PC_BITS) is allowed by program bus
        self.bitwise_lookup_bus
            .send_range(rd_data[0].clone(), rd_data[1].clone())
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[2].clone(), RV32_CELL_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[3].clone(), PC_BITS - RV32_CELL_BITS * 3)
            .eval(builder, is_valid);

        builder.assert_bool(imm_sign);

        // Constrain to_pc_least_sig_bit + 2 * to_pc_limbs = rs1 + imm as a i32 addition with 2
        // limbs RISC-V spec explicitly sets the least significant bit of `to_pc` to 0
        let rs1_limbs_01 = rs1[0] + rs1[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let rs1_limbs_23 = rs1[2] + rs1[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let inv = AB::F::from_canonical_u32(1 << 16).inverse();

        builder.assert_bool(to_pc_least_sig_bit);
        let carry = (rs1_limbs_01 + imm - to_pc_limbs[0] * AB::F::TWO - to_pc_least_sig_bit) * inv;
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
            to_pc_limbs[0] * AB::F::TWO + to_pc_limbs[1] * AB::F::from_canonical_u32(1 << 16);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JALR);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [rs1.map(|x| x.into())].into(),
            writes: [rd_data].into(),
            instruction: SignedImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm.into(),
                imm_sign: imm_sign.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32JalrOpcode::CLASS_OFFSET
    }
}

pub struct Rv32JalrStep<A> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A> Rv32JalrStep<A> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            adapter,
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, CTX, A> SingleTraceStep<F, CTX> for Rv32JalrStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [u8; RV32_REGISTER_NUM_LIMBS],
            WriteData = [u8; RV32_REGISTER_NUM_LIMBS],
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalrOpcode::from_usize(opcode - Rv32JalrOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        row_slice: &mut [F],
    ) -> Result<()> {
        let Instruction { opcode, c, g, .. } = *instruction;

        let local_opcode =
            Rv32JalrOpcode::from_usize(opcode.local_opcode_idx(Rv32JalrOpcode::CLASS_OFFSET));

        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let rs1 = self.adapter.read(state.memory, instruction, adapter_row);
        // TODO(ayush): avoid this conversion
        let rs1_val = compose(rs1.map(F::from_canonical_u8));

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        // TODO(ayush): this is bad since we're treating adapters as generic. maybe
        //              add a .state() function to adapters or get_from_pc like in air
        let adapter_row_ref: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();
        let from_pc = adapter_row_ref.from_state.pc.as_canonical_u32();

        let (to_pc, rd_data) = run_jalr(local_opcode, from_pc, imm_extended, rs1_val);

        let mask = (1 << 15) - 1;
        let to_pc_least_sig_bit = rs1_val.wrapping_add(imm_extended) & 1;

        let to_pc_limbs = array::from_fn(|i| ((to_pc >> (1 + i * 15)) & mask));

        let core_row: &mut Rv32JalrCoreCols<F> = core_row.borrow_mut();
        core_row.imm = c;
        core_row.rd_data = array::from_fn(|i| F::from_canonical_u32(rd_data[i + 1]));
        core_row.rs1_data = rs1.map(F::from_canonical_u8);
        core_row.to_pc_least_sig_bit = F::from_canonical_u32(to_pc_least_sig_bit);
        core_row.to_pc_limbs = to_pc_limbs.map(F::from_canonical_u32);
        core_row.imm_sign = g;
        core_row.is_valid = F::ONE;

        self.adapter.write(
            state.memory,
            instruction,
            adapter_row,
            &rd_data.map(|x| x as u8),
        );

        *state.pc = to_pc;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        let core_row: &mut Rv32JalrCoreCols<F> = core_row.borrow_mut();

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        // TODO(ayush): this shouldn't be here since it is generic on A
        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();

        // composed is the composition of 3 most significant limbs of rd
        let composed = core_row
            .rd_data
            .iter()
            .enumerate()
            .fold(F::ZERO, |acc, (i, &val)| {
                acc + val * F::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });

        let least_sig_limb =
            adapter_row.from_state.pc + F::from_canonical_u32(DEFAULT_PC_STEP) - composed;

        let rd_data: [F; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            if i == 0 {
                least_sig_limb
            } else {
                core_row.rd_data[i - 1]
            }
        });

        self.bitwise_lookup_chip
            .request_range(rd_data[0].as_canonical_u32(), rd_data[1].as_canonical_u32());

        self.range_checker_chip
            .add_count(rd_data[2].as_canonical_u32(), RV32_CELL_BITS);
        self.range_checker_chip
            .add_count(rd_data[3].as_canonical_u32(), PC_BITS - RV32_CELL_BITS * 3);

        self.range_checker_chip
            .add_count(core_row.to_pc_limbs[0].as_canonical_u32(), 15);
        self.range_checker_chip
            .add_count(core_row.to_pc_limbs[1].as_canonical_u32(), 14);
    }
}

impl<F, A> StepExecutorE1<F> for Rv32JalrStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<
            F,
            ReadData = [u8; RV32_REGISTER_NUM_LIMBS],
            WriteData = [u8; RV32_REGISTER_NUM_LIMBS],
        >,
{
    fn execute_e1<Mem, Ctx>(
        &mut self,
        state: VmStateMut<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Mem: GuestMemory,
    {
        let Instruction { opcode, c, g, .. } = instruction;

        let local_opcode =
            Rv32JalrOpcode::from_usize(opcode.local_opcode_idx(Rv32JalrOpcode::CLASS_OFFSET));

        let rs1 = self.adapter.read(state.memory, instruction);
        let rs1 = u32::from_le_bytes(rs1);

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        // TODO(ayush): should this be [u8; 4]?
        let (to_pc, rd) = run_jalr(local_opcode, *state.pc, imm_extended, rs1);
        let rd = rd.map(|x| x as u8);

        self.adapter.write(state.memory, instruction, &rd);

        *state.pc = to_pc;

        Ok(())
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
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
        array::from_fn(|i: usize| ((pc + DEFAULT_PC_STEP) >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX),
    )
}
