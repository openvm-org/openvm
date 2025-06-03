use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
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
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{Rv32RdWriteAdapterCols, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32AuipcCoreCols<T> {
    pub is_valid: T,
    // The limbs of the immediate except the least significant limb since it is always 0
    pub imm_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    // The limbs of the PC except the most significant and the least significant limbs
    pub pc_limbs: [T; RV32_REGISTER_NUM_LIMBS - 2],
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Debug, Clone, Copy, derive_new::new)]
pub struct Rv32AuipcCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32AuipcCoreAir {
    fn width(&self) -> usize {
        Rv32AuipcCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32AuipcCoreAir
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
        let cols: &Rv32AuipcCoreCols<AB::Var> = (*local_core).borrow();

        let Rv32AuipcCoreCols {
            is_valid,
            imm_limbs,
            pc_limbs,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);

        // We want to constrain rd = pc + imm (i32 add) where:
        // - rd_data represents limbs of rd
        // - pc_limbs are limbs of pc except the most and least significant limbs
        // - imm_limbs are limbs of imm except the least significant limb

        // We know that rd_data[0] is equal to the least significant limb of PC
        // Thus, the intermediate value will be equal to PC without its most significant limb:
        let intermed_val = rd_data[0]
            + pc_limbs
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                    acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
                });

        // Compute the most significant limb of PC
        let pc_msl = (from_pc - intermed_val)
            * AB::F::from_canonical_usize(1 << (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
                .inverse();

        // The vector pc_limbs contains the actual limbs of PC in little endian order
        let pc_limbs = [rd_data[0]]
            .iter()
            .chain(pc_limbs.iter())
            .map(|x| (*x).into())
            .chain([pc_msl])
            .collect::<Vec<AB::Expr>>();

        let mut carry: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << RV32_CELL_BITS).inverse();

        // Don't need to constrain the least significant limb of the addition
        // since we already know that rd_data[0] = pc_limbs[0] and the least significant limb of imm
        // is 0 Note: imm_limbs doesn't include the least significant limb so imm_limbs[i -
        // 1] means the i-th limb of imm
        for i in 1..RV32_REGISTER_NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (pc_limbs[i].clone() + imm_limbs[i - 1] - rd_data[i] + carry[i - 1].clone());
            builder.when(is_valid).assert_bool(carry[i].clone());
        }

        // Range checking of rd_data entries to RV32_CELL_BITS bits
        for i in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
            self.bus
                .send_range(rd_data[i * 2], rd_data[i * 2 + 1])
                .eval(builder, is_valid);
        }

        // The immediate and PC limbs need range checking to ensure they're within [0,
        // 2^RV32_CELL_BITS) Since we range check two items at a time, doing this way helps
        // efficiently divide the limbs into groups of 2 Note: range checking the limbs of
        // immediate and PC separately would result in additional range checks       since
        // they both have odd number of limbs that need to be range checked
        let mut need_range_check: Vec<AB::Expr> = Vec::new();
        for limb in imm_limbs {
            need_range_check.push(limb.into());
        }

        assert_eq!(pc_limbs.len(), RV32_REGISTER_NUM_LIMBS);
        // use enumerate to match pc_limbs[0] => i = 0, pc_limbs[1] => i = 1, ...
        // pc_limbs[0] is already range checked through rd_data[0], so we skip it
        for (i, limb) in pc_limbs.iter().enumerate().skip(1) {
            // the most significant limb is pc_limbs[3] => i = 3
            if i == pc_limbs.len() - 1 {
                // Range check the most significant limb of pc to be in [0,
                // 2^{PC_BITS-(RV32_REGISTER_NUM_LIMBS-1)*RV32_CELL_BITS})
                need_range_check.push(
                    (*limb).clone()
                        * AB::Expr::from_canonical_usize(
                            1 << (pc_limbs.len() * RV32_CELL_BITS - PC_BITS),
                        ),
                );
            } else {
                need_range_check.push((*limb).clone());
            }
        }

        // need_range_check contains (RV32_REGISTER_NUM_LIMBS - 1) elements from imm_limbs
        // and (RV32_REGISTER_NUM_LIMBS - 1) elements from pc_limbs
        // Hence, is of even length 2*RV32_REGISTER_NUM_LIMBS - 2
        assert_eq!(need_range_check.len() % 2, 0);
        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0].clone(), pair[1].clone())
                .eval(builder, is_valid);
        }

        let imm = imm_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [rd_data.map(|x| x.into())].into(),
            instruction: ImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32AuipcOpcode::CLASS_OFFSET
    }
}

pub struct Rv32AuipcCoreStep<A> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<A> Rv32AuipcCoreStep<A> {
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

impl<F, CTX, A> TraceStep<F, CTX> for Rv32AuipcCoreStep<A>
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
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", AUIPC)
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let Instruction { opcode, c: imm, .. } = instruction;

        let local_opcode =
            Rv32AuipcOpcode::from_usize(opcode.local_opcode_idx(Rv32AuipcOpcode::CLASS_OFFSET));

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let imm_u32 = imm.as_canonical_u32();
        let rd = run_auipc(local_opcode, *state.pc, imm_u32);

        let core_row: &mut Rv32AuipcCoreCols<F> = core_row.borrow_mut();
        core_row.rd_data = rd.map(F::from_canonical_u8);

        // TODO(ayush): see if there's a better way
        // We decompose during fill_trace_row later:
        core_row.imm_limbs[0] = *imm;

        self.adapter
            .write(state.memory, instruction, adapter_row, &rd);

        // TODO(ayush): add increment_pc function to vmstate
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        let core_row: &mut Rv32AuipcCoreCols<F> = core_row.borrow_mut();

        core_row.is_valid = F::ONE;

        // TODO(ayush): this is bad since we're treating adapters as generic. maybe
        //              add a .state() function to adapters or get_from_pc like in air
        let adapter_row: &mut Rv32RdWriteAdapterCols<F> = adapter_row.borrow_mut();
        let from_pc = adapter_row.from_state.pc.as_canonical_u32();

        let pc_limbs = from_pc.to_le_bytes();
        let imm = core_row.imm_limbs[0].as_canonical_u32();
        let imm_limbs = imm.to_le_bytes();
        debug_assert_eq!(imm_limbs[3], 0);
        core_row.imm_limbs = from_fn(|i| F::from_canonical_u8(imm_limbs[i]));
        // only the middle 2 limbs:
        core_row.pc_limbs = from_fn(|i| F::from_canonical_u8(pc_limbs[i + 1]));

        // range checks:
        let rd_data = core_row.rd_data.map(|x| x.as_canonical_u32());
        for pair in rd_data.chunks_exact(2) {
            self.bitwise_lookup_chip.request_range(pair[0], pair[1]);
        }
        // hardcoding for performance: first 3 limbs of imm_limbs, last 3 limbs of pc_limbs where
        // most significant limb of pc_limbs is shifted up
        self.bitwise_lookup_chip
            .request_range(imm_limbs[0] as u32, imm_limbs[1] as u32);
        self.bitwise_lookup_chip
            .request_range(imm_limbs[2] as u32, pc_limbs[1] as u32);
        let msl_shift = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - PC_BITS;
        self.bitwise_lookup_chip
            .request_range(pc_limbs[2] as u32, (pc_limbs[3] as u32) << msl_shift);
    }
}

impl<F, A> StepExecutorE1<F> for Rv32AuipcCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = (), WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, c: imm, .. } = instruction;

        let local_opcode =
            Rv32AuipcOpcode::from_usize(opcode.local_opcode_idx(Rv32AuipcOpcode::CLASS_OFFSET));

        let imm = imm.as_canonical_u32();
        let rd = run_auipc(local_opcode, *state.pc, imm);

        self.adapter.write(state, instruction, &rd);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

// returns rd_data
// TODO(ayush): remove _opcode
#[inline(always)]
pub(super) fn run_auipc(
    _opcode: Rv32AuipcOpcode,
    pc: u32,
    imm: u32,
) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    rd.to_le_bytes()
}
