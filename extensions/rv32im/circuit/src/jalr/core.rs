use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterTraceFiller, AdapterTraceStep,
        E2PreCompute, EmptyAdapterCoreLayout, ExecuteFunc,
        ExecutionError::InvalidInstruction,
        RecordArena, Result, SignedImmInstruction, StepExecutorE1, StepExecutorE2, TraceFiller,
        TraceStep, VmAdapterInterface, VmCoreAir, VmSegmentState, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_REGISTER_AS,
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalrOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32JalrCoreRecord {
    pub imm: u16,
    pub from_pc: u32,
    pub rs1_val: u32,
    pub imm_sign: bool,
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

impl<F, CTX, A> TraceStep<F, CTX> for Rv32JalrStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [u8; RV32_REGISTER_NUM_LIMBS],
            WriteData = [u8; RV32_REGISTER_NUM_LIMBS],
        >,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (A::RecordMut<'a>, &'a mut Rv32JalrCoreRecord);

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalrOpcode::from_usize(opcode - Rv32JalrOpcode::CLASS_OFFSET)
        )
    }

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let Instruction { opcode, c, g, .. } = *instruction;

        debug_assert_eq!(
            opcode.local_opcode_idx(Rv32JalrOpcode::CLASS_OFFSET),
            JALR as usize
        );

        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        core_record.rs1_val = u32::from_le_bytes(self.adapter.read(
            state.memory,
            instruction,
            &mut adapter_record,
        ));

        core_record.imm = c.as_canonical_u32() as u16;
        core_record.imm_sign = g.is_one();
        core_record.from_pc = *state.pc;

        let (to_pc, rd_data) = run_jalr(
            core_record.from_pc,
            core_record.rs1_val,
            core_record.imm,
            core_record.imm_sign,
        );

        self.adapter
            .write(state.memory, instruction, rd_data, &mut adapter_record);

        // RISC-V spec explicitly sets the least significant bit of `to_pc` to 0
        *state.pc = to_pc & !1;

        Ok(())
    }
}
impl<F, CTX, A> TraceFiller<F, CTX> for Rv32JalrStep<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F, CTX>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &Rv32JalrCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv32JalrCoreCols<F> = core_row.borrow_mut();

        let (to_pc, rd_data) =
            run_jalr(record.from_pc, record.rs1_val, record.imm, record.imm_sign);
        let to_pc_limbs = [(to_pc & ((1 << 16) - 1)) >> 1, to_pc >> 16];
        self.range_checker_chip.add_count(to_pc_limbs[0], 15);
        self.range_checker_chip
            .add_count(to_pc_limbs[1], PC_BITS - 16);
        self.bitwise_lookup_chip
            .request_range(rd_data[0] as u32, rd_data[1] as u32);

        self.range_checker_chip
            .add_count(rd_data[2] as u32, RV32_CELL_BITS);
        self.range_checker_chip
            .add_count(rd_data[3] as u32, PC_BITS - RV32_CELL_BITS * 3);

        // Write in reverse order
        core_row.imm_sign = F::from_bool(record.imm_sign);
        core_row.to_pc_limbs = to_pc_limbs.map(F::from_canonical_u32);
        core_row.to_pc_least_sig_bit = F::from_bool(to_pc & 1 == 1);
        // fill_trace_row is called only on valid rows
        core_row.is_valid = F::ONE;
        core_row.rs1_data = record.rs1_val.to_le_bytes().map(F::from_canonical_u8);
        core_row
            .rd_data
            .iter_mut()
            .rev()
            .zip(rd_data.iter().skip(1).rev())
            .for_each(|(dst, src)| {
                *dst = F::from_canonical_u8(*src);
            });
        core_row.imm = F::from_canonical_u16(record.imm);
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalrPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<F, A> StepExecutorE1<F> for Rv32JalrStep<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalrPreCompute>()
    }
    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>> {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        let fn_ptr = if enabled {
            execute_e1_impl::<_, _, true>
        } else {
            execute_e1_impl::<_, _, false>
        };
        Ok(fn_ptr)
    }
}

impl<F, A> StepExecutorE2<F> for Rv32JalrStep<A>
where
    F: PrimeField32,
{
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalrPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let fn_ptr = if enabled {
            execute_e2_impl::<_, _, true>
        } else {
            execute_e2_impl::<_, _, false>
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx, const ENABLED: bool>(
    pre_compute: &JalrPreCompute,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let rs1 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let to_pc = rs1.wrapping_add(pre_compute.imm_extended);
    let to_pc = to_pc - (to_pc & 1);
    debug_assert!(to_pc < (1 << PC_BITS));
    let rd = (vm_state.pc + DEFAULT_PC_STEP).to_le_bytes();

    if ENABLED {
        vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    }

    vm_state.pc = to_pc;
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, const ENABLED: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &JalrPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, ENABLED>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx, const ENABLED: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &E2PreCompute<JalrPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, ENABLED>(&pre_compute.data, vm_state);
}

impl<A> Rv32JalrStep<A> {
    /// Return true if enabled.
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut JalrPreCompute,
    ) -> Result<bool> {
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(InvalidInstruction(pc));
        }
        *data = JalrPreCompute {
            imm_extended,
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok(enabled)
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jalr(pc: u32, rs1: u32, imm: u16, imm_sign: bool) -> (u32, [u8; 4]) {
    let to_pc = rs1.wrapping_add(imm as u32 + (imm_sign as u32 * 0xffff0000));
    assert!(to_pc < (1 << PC_BITS));
    (to_pc, pc.wrapping_add(DEFAULT_PC_STEP).to_le_bytes())
}
