use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, ImmInstruction, Result,
        StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, RV_J_TYPE_IMM_BITS};

const ADDITIONAL_BITS: u32 = 0b11000000;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalLuiCoreCols<T> {
    pub imm: T,
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_jal: T,
    pub is_lui: T,
}

#[derive(Debug, Clone, Copy, derive_new::new)]
pub struct Rv32JalLuiCoreAir {
    pub bus: BitwiseOperationLookupBus,
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
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.when(is_lui).assert_zero(rd[0]);

        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.bus
                .send_range(rd[i * 2], rd[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        // In case of JAL constrain that last limb has at most [last_limb_bits] bits

        let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1 << x));
        let additional_bits = AB::F::from_canonical_u32(additional_bits);
        self.bus
            .send_xor(rd[3], additional_bits, rd[3] + additional_bits)
            .eval(builder, is_jal);

        let intermed_val = rd
            .iter()
            .skip(1)
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });

        // Constrain that imm * 2^4 is the correct composition of intermed_val in case of LUI
        builder.when(is_lui).assert_eq(
            intermed_val.clone(),
            imm * AB::F::from_canonical_u32(1 << (12 - RV32_CELL_BITS)),
        );

        let intermed_val = rd[0] + intermed_val * AB::Expr::from_canonical_u32(1 << RV32_CELL_BITS);
        // Constrain that from_pc + DEFAULT_PC_STEP is the correct composition of intermed_val in
        // case of JAL
        builder.when(is_jal).assert_eq(
            intermed_val,
            from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP),
        );

        let to_pc = from_pc + is_lui * AB::F::from_canonical_u32(DEFAULT_PC_STEP) + is_jal * imm;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_canonical_u32(LUI as u32)
                + is_jal * AB::F::from_canonical_u32(JAL as u32),
        );

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

    fn start_offset(&self) -> usize {
        Rv32JalLuiOpcode::CLASS_OFFSET
    }
}

pub struct Rv32JalLuiStep<A> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<A> Rv32JalLuiStep<A> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        Self {
            adapter,
            bitwise_lookup_chip,
        }
    }
}

impl<F, CTX, A> TraceStep<F, CTX> for Rv32JalLuiStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = (),
            WriteData = [u8; RV32_REGISTER_NUM_LIMBS],
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalLuiOpcode::from_usize(opcode - Rv32JalLuiOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let Instruction { opcode, c: imm, .. } = instruction;

        let local_opcode =
            Rv32JalLuiOpcode::from_usize(opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET));

        let mut row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let core_row: &mut Rv32JalLuiCoreCols<F> = core_row.borrow_mut();

        // `c` can be "negative" as a field element
        let imm_f = imm.as_canonical_u32();
        let signed_imm = match local_opcode {
            JAL => {
                if imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)) {
                    imm_f as i32
                } else {
                    let neg_imm_f = F::ORDER_U32 - imm_f;
                    debug_assert!(neg_imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)));
                    -(neg_imm_f as i32)
                }
            }
            LUI => imm_f as i32,
        };
        let (to_pc, rd_data) = run_jal_lui(local_opcode, *state.pc, signed_imm);

        core_row.rd_data = rd_data.map(F::from_canonical_u8);
        core_row.imm = instruction.c;
        core_row.is_jal = F::from_bool(local_opcode == JAL);
        core_row.is_lui = F::from_bool(local_opcode == LUI);

        self.adapter
            .write(state.memory, instruction, adapter_row, &rd_data);

        *state.pc = to_pc;

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        let core_row: &mut Rv32JalLuiCoreCols<F> = core_row.borrow_mut();

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        let rd_data = core_row.rd_data.map(|x| x.as_canonical_u32());
        for pair in rd_data.chunks_exact(2) {
            self.bitwise_lookup_chip.request_range(pair[0], pair[1]);
        }
        if core_row.is_jal == F::ONE {
            self.bitwise_lookup_chip
                .request_xor(rd_data[3], ADDITIONAL_BITS);
        }
    }
}

impl<F, A> StepExecutorE1<F> for Rv32JalLuiStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = (), WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
{
    fn execute_e1<Mem, Ctx>(
        &mut self,
        state: VmStateMut<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Mem: GuestMemory,
    {
        let Instruction { opcode, c: imm, .. } = instruction;

        let local_opcode =
            Rv32JalLuiOpcode::from_usize(opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET));

        let imm_f = imm.as_canonical_u32();
        let signed_imm = match local_opcode {
            JAL => {
                if imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)) {
                    imm_f as i32
                } else {
                    let neg_imm_f = F::ORDER_U32 - imm_f;
                    debug_assert!(neg_imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)));
                    -(neg_imm_f as i32)
                }
            }
            LUI => imm_f as i32,
        };
        let (to_pc, rd) = run_jal_lui(local_opcode, *state.pc, signed_imm);

        self.adapter.write(state.memory, instruction, &rd);

        *state.pc = to_pc;

        Ok(())
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jal_lui(
    opcode: Rv32JalLuiOpcode,
    pc: u32,
    imm: i32,
) -> (u32, [u8; RV32_REGISTER_NUM_LIMBS]) {
    match opcode {
        JAL => {
            let rd_data = (pc + DEFAULT_PC_STEP).to_le_bytes();
            let next_pc = pc as i32 + imm;
            assert!(next_pc >= 0);
            (next_pc as u32, rd_data)
        }
        LUI => {
            let imm = imm as u32;
            let rd = imm << 12;
            (pc + DEFAULT_PC_STEP, rd.to_le_bytes())
        }
    }
}

// TODO(ayush): move from here
#[test]
fn test_additional_bits() {
    let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
    let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1u32 << x));
    assert_eq!(additional_bits, ADDITIONAL_BITS);
}
