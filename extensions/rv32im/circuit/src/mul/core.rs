use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, EmptyAdapterCoreLayout, InsExecutorE1, InstructionExecutor,
        MinimalInstruction, RecordArena, Result, TraceFiller, VmAdapterInterface, VmCoreAir,
        VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MultiplicationCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct MultiplicationCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MultiplicationCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &MultiplicationCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Define carry[i] = (sum_{k=0}^{i} b[k] * c[i - k] + carry[i - 1] - a[i]) / 2^LIMB_BITS.
        // If 0 <= a[i], carry[i] < 2^LIMB_BITS, it can be proven that a[i] = sum_{k=0}^{i} (b[k] *
        // c[i - k]) % 2^LIMB_BITS as necessary.
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_u32(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            let expected_limb = if i == 0 {
                AB::Expr::ZERO
            } else {
                carry[i - 1].clone()
            } + (0..=i).fold(AB::Expr::ZERO, |acc, k| acc + (b[k] * c[i - k]));
            carry[i] = AB::Expr::from(carry_divide) * (expected_limb - a[i]);
        }

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.bus
                .send(vec![(*a).into(), carry.clone()])
                .eval(builder, cols.is_valid);
        }

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, MulOpcode::MUL);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MultiplicationCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct MultiplicationStep<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct MultiplicationFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MultiplicationFiller<A, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        adapter: A,
        range_tuple_chip: SharedRangeTupleCheckerChip<2>,
        offset: usize,
    ) -> Self {
        // The RangeTupleChecker is used to range check (a[i], carry[i]) pairs where 0 <= i
        // < NUM_LIMBS. a[i] must have LIMB_BITS bits and carry[i] is the sum of i + 1 bytes
        // (with LIMB_BITS bits).
        debug_assert!(
            range_tuple_chip.sizes()[0] == 1 << LIMB_BITS,
            "First element of RangeTupleChecker must have size {}",
            1 << LIMB_BITS
        );
        debug_assert!(
            range_tuple_chip.sizes()[1] >= (1 << LIMB_BITS) * NUM_LIMBS as u32,
            "Second element of RangeTupleChecker must have size of at least {}",
            (1 << LIMB_BITS) * NUM_LIMBS as u32
        );

        Self {
            adapter,
            offset,
            range_tuple_chip,
        }
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<F, RA>
    for MultiplicationStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceStep<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut MultiplicationCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", MulOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let Instruction { opcode, .. } = instruction;

        debug_assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.offset)),
            MulOpcode::MUL
        );
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (a, _) = run_mul::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;

        self.adapter
            .write(state.memory, instruction, [a].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for MultiplicationFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &MultiplicationCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut MultiplicationCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let (a, carry) = run_mul::<NUM_LIMBS, LIMB_BITS>(&record.b, &record.c);

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.range_tuple_chip.add_count(&[*a as u32, *carry]);
        }

        // write in reverse order
        core_row.is_valid = F::ONE;
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = a.map(F::from_canonical_u8);
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MultiPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F, A, const LIMB_BITS: usize> InsExecutorE1<F>
    for MultiplicationStep<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<MultiPreCompute>()
    }
    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let pre_compute: &mut MultiPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_impl)
    }
}

impl<F, A, const LIMB_BITS: usize> InsExecutorE2<F>
    for MultiplicationStep<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MultiPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<MultiPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_impl)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &MultiPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = rs1.wrapping_mul(rs2);
    vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd.to_le_bytes());

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MultiPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MultiPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, vm_state);
}

impl<A, const LIMB_BITS: usize> MultiplicationStep<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MultiPreCompute,
    ) -> Result<()> {
        assert_eq!(
            MulOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset)),
            MulOpcode::MUL
        );
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(ExecutionError::InvalidInstruction(pc));
        }

        *data = MultiPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

// returns mul, carry
#[inline(always)]
pub(super) fn run_mul<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], [u32; NUM_LIMBS]) {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut res = 0u32;
        if i > 0 {
            res = carry[i - 1];
        }
        for j in 0..=i {
            res += (x[j] as u32) * (y[i - j] as u32);
        }
        carry[i] = res >> LIMB_BITS;
        res %= 1u32 << LIMB_BITS;
        result[i] = res as u8;
    }
    (result, carry)
}
