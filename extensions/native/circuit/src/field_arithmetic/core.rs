use std::borrow::{Borrow, BorrowMut};

use itertools::izip;
use openvm_circuit::{
    arch::{
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterTraceFiller, AdapterTraceStep,
        E2PreCompute, EmptyAdapterCoreLayout, ExecuteFunc, ExecutionError, InsExecutorE1,
        InsExecutorE2, InstructionExecutor, MinimalInstruction, RecordArena, TraceFiller,
        VmAdapterInterface, VmCoreAir, VmSegmentState, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, LocalOpcode,
};
use openvm_native_compiler::{
    conversion::AS,
    FieldArithmeticOpcode::{self, *},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FieldArithmeticCoreCols<T> {
    pub a: T,
    pub b: T,
    pub c: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,
}

#[derive(derive_new::new, Copy, Clone, Debug)]
pub struct FieldArithmeticCoreAir {}

impl<F: Field> BaseAir<F> for FieldArithmeticCoreAir {
    fn width(&self) -> usize {
        FieldArithmeticCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldArithmeticCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FieldArithmeticCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 2]>,
    I::Writes: From<[[AB::Expr; 1]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FieldArithmeticCoreCols<_> = local_core.borrow();

        let a = cols.a;
        let b = cols.b;
        let c = cols.c;

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [ADD, SUB, MUL, DIV];
        let results = [b + c, b - c, b * c, b * cols.divisor_inv];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative
        //   inverse of `io.y`.

        let mut is_valid = AB::Expr::ZERO;
        let mut expected_opcode = AB::Expr::ZERO;
        let mut expected_result = AB::Expr::ZERO;
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        builder.assert_eq(a, expected_result);
        builder.assert_bool(is_valid.clone());
        builder.assert_eq(cols.is_div, c * cols.divisor_inv);

        AdapterAirContext {
            to_pc: None,
            reads: [[cols.b.into()], [cols.c.into()]].into(),
            writes: [[cols.a.into()]].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        FieldArithmeticOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FieldArithmeticRecord<F> {
    pub b: F,
    pub c: F,
    pub local_opcode: u8,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct FieldArithmeticCoreStep<A> {
    adapter: A,
}

#[derive(derive_new::new)]
pub struct FieldArithmeticCoreFiller<A> {
    adapter: A,
}

impl<F, A, RA> InstructionExecutor<F, RA> for FieldArithmeticCoreStep<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceStep<F, ReadData = [F; 2], WriteData = [F; 1]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FieldArithmeticRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - FieldArithmeticOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        core_record.local_opcode =
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET) as u8;

        let opcode = FieldArithmeticOpcode::from_usize(core_record.local_opcode as usize);
        let a_val = run_field_arithmetic(opcode, core_record.b, core_record.c);

        self.adapter
            .write(state.memory, instruction, [a_val], &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for FieldArithmeticCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FieldArithmeticRecord<F> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FieldArithmeticCoreCols<_> = core_row.borrow_mut();

        let opcode = FieldArithmeticOpcode::from_usize(record.local_opcode as usize);
        let result = run_field_arithmetic(opcode, record.b, record.c);

        // Writing in reverse order to avoid overwriting the `record`
        core_row.divisor_inv = if opcode == FieldArithmeticOpcode::DIV {
            record.c.inverse()
        } else {
            F::ZERO
        };

        core_row.is_div = F::from_bool(opcode == FieldArithmeticOpcode::DIV);
        core_row.is_mul = F::from_bool(opcode == FieldArithmeticOpcode::MUL);
        core_row.is_sub = F::from_bool(opcode == FieldArithmeticOpcode::SUB);
        core_row.is_add = F::from_bool(opcode == FieldArithmeticOpcode::ADD);

        core_row.c = record.c;
        core_row.b = record.b;
        core_row.a = result;
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldArithmeticPreCompute {
    a: u32,
    b_or_imm: u32,
    c_or_imm: u32,
    e: u32,
    f: u32,
}

impl<A> FieldArithmeticCoreStep<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldArithmeticPreCompute,
    ) -> Result<(bool, bool, FieldArithmeticOpcode), ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            e,
            f,
            ..
        } = inst;

        let local_opcode = FieldArithmeticOpcode::from_usize(
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET),
        );

        let a = a.as_canonical_u32();
        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let a_is_imm = e == RV32_IMM_AS;
        let b_is_imm = f == RV32_IMM_AS;

        let b_or_imm = if a_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };
        let c_or_imm = if b_is_imm {
            transmute_field_to_u32(&c)
        } else {
            c.as_canonical_u32()
        };

        *data = FieldArithmeticPreCompute {
            a,
            b_or_imm,
            c_or_imm,
            e,
            f,
        };

        Ok((a_is_imm, b_is_imm, local_opcode))
    }
}

impl<F, A> InsExecutorE1<F> for FieldArithmeticCoreStep<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<FieldArithmeticPreCompute>()
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, ExecutionError> {
        let pre_compute: &mut FieldArithmeticPreCompute = data.borrow_mut();

        let (a_is_imm, b_is_imm, local_opcode) = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match (local_opcode, a_is_imm, b_is_imm) {
            (FieldArithmeticOpcode::ADD, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
        };

        Ok(fn_ptr)
    }
}

impl<F, A> InsExecutorE2<F> for FieldArithmeticCoreStep<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<FieldArithmeticPreCompute>>()
    }

    #[inline(always)]
    fn pre_compute_e2<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, ExecutionError> {
        let pre_compute: &mut E2PreCompute<FieldArithmeticPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (a_is_imm, b_is_imm, local_opcode) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = match (local_opcode, a_is_imm, b_is_imm) {
            (FieldArithmeticOpcode::ADD, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
        };

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldArithmeticPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, A_IS_IMM, B_IS_IMM, OPCODE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldArithmeticPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, A_IS_IMM, B_IS_IMM, OPCODE>(&pre_compute.data, vm_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &FieldArithmeticPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read values based on the adapter logic
    let b_val = if A_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(pre_compute.e, pre_compute.b_or_imm)[0]
    };
    let c_val = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.c_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(pre_compute.f, pre_compute.c_or_imm)[0]
    };

    let a_val = match OPCODE {
        0 => b_val + c_val, // ADD
        1 => b_val - c_val, // SUB
        2 => b_val * c_val, // MUL
        3 => {
            // DIV
            if c_val.is_zero() {
                vm_state.exit_code = Err(ExecutionError::Fail { pc: vm_state.pc });
                return;
            }
            b_val * c_val.inverse()
        }
        _ => panic!("Invalid field arithmetic opcode: {OPCODE}"),
    };

    vm_state.vm_write::<F, 1>(AS::Native as u32, pre_compute.a, &[a_val]);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

pub(super) fn run_field_arithmetic<F: Field>(opcode: FieldArithmeticOpcode, b: F, c: F) -> F {
    match opcode {
        FieldArithmeticOpcode::ADD => b + c,
        FieldArithmeticOpcode::SUB => b - c,
        FieldArithmeticOpcode::MUL => b * c,
        FieldArithmeticOpcode::DIV => {
            assert!(!c.is_zero(), "Division by zero");
            b * c.inverse()
        }
    }
}
