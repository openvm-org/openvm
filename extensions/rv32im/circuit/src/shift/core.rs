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
        MinimalInstruction, RecordArena, Result, StepExecutorE1, StepExecutorE2, TraceFiller,
        TraceStep, VmAdapterInterface, VmCoreAir, VmSegmentState, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::imm_to_bytes;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct ShiftCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_sll_flag: T,
    pub opcode_srl_flag: T,
    pub opcode_sra_flag: T,

    // bit_multiplier = 2^bit_shift
    pub bit_multiplier_left: T,
    pub bit_multiplier_right: T,

    // Sign of x for SRA
    pub b_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Part of each x[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; NUM_LIMBS],
}

/// RV32 shift AIR.
/// Note: when the shift amount from operand is greater than the number of bits, only shift
/// `shift_amount % num_bits` bits. This matches the RV32 specs for SLL/SRL/SRA.
#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct ShiftCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_sll_flag,
            cols.opcode_srl_flag,
            cols.opcode_sra_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;
        let right_shift = cols.opcode_srl_flag + cols.opcode_sra_flag;

        // Constrain that bit_shift, bit_multiplier are correct, i.e. that bit_multiplier =
        // 1 << bit_shift. Because the sum of all bit_shift_marker[i] is constrained to be
        // 1, bit_shift is guaranteed to be in range.
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;

        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            bit_marker_sum += cols.bit_shift_marker[i].into();
            bit_shift += AB::Expr::from_canonical_usize(i) * cols.bit_shift_marker[i];

            let mut when_bit_shift = builder.when(cols.bit_shift_marker[i]);
            when_bit_shift.assert_eq(
                cols.bit_multiplier_left,
                AB::Expr::from_canonical_usize(1 << i) * cols.opcode_sll_flag,
            );
            when_bit_shift.assert_eq(
                cols.bit_multiplier_right,
                AB::Expr::from_canonical_usize(1 << i) * right_shift.clone(),
            );
        }
        builder.when(is_valid.clone()).assert_one(bit_marker_sum);

        // Check that a[i] = b[i] <</>> c[i] both on the bit and limb shift level if c <
        // NUM_LIMBS * LIMB_BITS.
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_canonical_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for j in 0..NUM_LIMBS {
                // SLL constraints
                if j < i {
                    when_limb_shift.assert_zero(a[j] * cols.opcode_sll_flag);
                } else {
                    let expected_a_left = if j - i == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j - i - 1].into() * cols.opcode_sll_flag
                    } + b[j - i] * cols.bit_multiplier_left
                        - AB::Expr::from_canonical_usize(1 << LIMB_BITS)
                            * cols.bit_shift_carry[j - i]
                            * cols.opcode_sll_flag;
                    when_limb_shift.assert_eq(a[j] * cols.opcode_sll_flag, expected_a_left);
                }

                // SRL and SRA constraints. Combining with above would require an additional column.
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        a[j] * right_shift.clone(),
                        cols.b_sign * AB::F::from_canonical_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    let expected_a_right = if j + i == NUM_LIMBS - 1 {
                        cols.b_sign * (cols.bit_multiplier_right - AB::F::ONE)
                    } else {
                        cols.bit_shift_carry[j + i + 1].into() * right_shift.clone()
                    } * AB::F::from_canonical_usize(1 << LIMB_BITS)
                        + right_shift.clone() * (b[j + i] - cols.bit_shift_carry[j + i]);
                    when_limb_shift.assert_eq(a[j] * cols.bit_multiplier_right, expected_a_right);
                }
            }
        }
        builder.when(is_valid.clone()).assert_one(limb_marker_sum);

        // Check that bit_shift and limb_shift are correct.
        let num_bits = AB::F::from_canonical_usize(NUM_LIMBS * LIMB_BITS);
        self.range_bus
            .range_check(
                (c[0] - limb_shift * AB::F::from_canonical_usize(LIMB_BITS) - bit_shift.clone())
                    * num_bits.inverse(),
                LIMB_BITS - ((NUM_LIMBS * LIMB_BITS) as u32).ilog2() as usize,
            )
            .eval(builder, is_valid.clone());

        // Check b_sign & b[NUM_LIMBS - 1] == b_sign using XOR
        builder.assert_bool(cols.b_sign);
        builder
            .when(not(cols.opcode_sra_flag))
            .assert_zero(cols.b_sign);

        let mask = AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        let b_sign_shifted = cols.b_sign * mask;
        self.bitwise_lookup_bus
            .send_xor(
                b[NUM_LIMBS - 1],
                mask,
                b[NUM_LIMBS - 1] + mask - (AB::Expr::from_canonical_u32(2) * b_sign_shifted),
            )
            .eval(builder, cols.opcode_sra_flag);

        for i in 0..(NUM_LIMBS / 2) {
            self.bitwise_lookup_bus
                .send_range(a[i * 2], a[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        for carry in cols.bit_shift_carry {
            self.range_bus
                .send(carry, bit_shift.clone())
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags
                .iter()
                .zip(ShiftOpcode::iter())
                .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
                }),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
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
pub struct ShiftCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

pub struct ShiftStep<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftStep<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
        offset: usize,
    ) -> Self {
        assert_eq!(NUM_LIMBS % 2, 0, "Number of limbs must be divisible by 2");
        Self {
            adapter,
            offset,
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, CTX, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceStep<F, CTX>
    for ShiftStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (
        A::RecordMut<'a>,
        &'a mut ShiftCoreRecord<NUM_LIMBS, LIMB_BITS>,
    );

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.offset))
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
        let Instruction { opcode, .. } = instruction;

        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (output, _, _) = run_shift::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = local_opcode as u8;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, CTX, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F, CTX>
    for ShiftStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F, CTX>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &ShiftCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut ShiftCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let opcode = ShiftOpcode::from_usize(record.local_opcode as usize);
        let (a, limb_shift, bit_shift) =
            run_shift::<NUM_LIMBS, LIMB_BITS>(opcode, &record.b, &record.c);

        for pair in a.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }

        let num_bits_log = (NUM_LIMBS * LIMB_BITS).ilog2();
        self.range_checker_chip.add_count(
            ((record.c[0] as usize - bit_shift - limb_shift * LIMB_BITS) >> num_bits_log) as u32,
            LIMB_BITS - num_bits_log as usize,
        );

        core_row.bit_shift_carry = if bit_shift == 0 {
            for _ in 0..NUM_LIMBS {
                self.range_checker_chip.add_count(0, 0);
            }
            [F::ZERO; NUM_LIMBS]
        } else {
            array::from_fn(|i| {
                let carry = match opcode {
                    ShiftOpcode::SLL => record.b[i] >> (LIMB_BITS - bit_shift),
                    _ => record.b[i] % (1 << bit_shift),
                };
                self.range_checker_chip.add_count(carry as u32, bit_shift);
                F::from_canonical_u8(carry)
            })
        };

        core_row.limb_shift_marker = [F::ZERO; NUM_LIMBS];
        core_row.limb_shift_marker[limb_shift] = F::ONE;
        core_row.bit_shift_marker = [F::ZERO; LIMB_BITS];
        core_row.bit_shift_marker[bit_shift] = F::ONE;

        core_row.b_sign = F::ZERO;
        if opcode == ShiftOpcode::SRA {
            core_row.b_sign = F::from_canonical_u8(record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1));
            self.bitwise_lookup_chip
                .request_xor(record.b[NUM_LIMBS - 1] as u32, 1 << (LIMB_BITS - 1));
        }

        core_row.bit_multiplier_right = match opcode {
            ShiftOpcode::SLL => F::ZERO,
            _ => F::from_canonical_usize(1 << bit_shift),
        };
        core_row.bit_multiplier_left = match opcode {
            ShiftOpcode::SLL => F::from_canonical_usize(1 << bit_shift),
            _ => F::ZERO,
        };

        core_row.opcode_sra_flag = F::from_bool(opcode == ShiftOpcode::SRA);
        core_row.opcode_srl_flag = F::from_bool(opcode == ShiftOpcode::SRL);
        core_row.opcode_sll_flag = F::from_bool(opcode == ShiftOpcode::SLL);

        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = a.map(F::from_canonical_u8);
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> StepExecutorE1<F>
    for ShiftStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
    }
    #[inline(always)]
    fn pre_compute_align(&self) -> usize {
        align_of::<ShiftPreCompute>()
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>> {
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        let fn_ptr = match (is_imm, shift_opcode) {
            (true, ShiftOpcode::SLL) => execute_e1_impl::<_, _, true, SllOp>,
            (false, ShiftOpcode::SLL) => execute_e1_impl::<_, _, false, SllOp>,
            (true, ShiftOpcode::SRL) => execute_e1_impl::<_, _, true, SrlOp>,
            (false, ShiftOpcode::SRL) => execute_e1_impl::<_, _, false, SrlOp>,
            (true, ShiftOpcode::SRA) => execute_e1_impl::<_, _, true, SraOp>,
            (false, ShiftOpcode::SRA) => execute_e1_impl::<_, _, false, SraOp>,
        };
        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> StepExecutorE2<F>
    for ShiftStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftPreCompute>>()
    }
    #[inline(always)]
    fn e2_pre_compute_align(&self) -> usize {
        align_of::<E2PreCompute<ShiftPreCompute>>()
    }

    #[inline(always)]
    fn pre_compute_e2<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>> {
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u16;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        let fn_ptr = match (is_imm, shift_opcode) {
            (true, ShiftOpcode::SLL) => execute_e2_impl::<_, _, true, SllOp>,
            (false, ShiftOpcode::SLL) => execute_e2_impl::<_, _, false, SllOp>,
            (true, ShiftOpcode::SRL) => execute_e2_impl::<_, _, true, SrlOp>,
            (false, ShiftOpcode::SRL) => execute_e2_impl::<_, _, false, SrlOp>,
            (true, ShiftOpcode::SRA) => execute_e2_impl::<_, _, true, SraOp>,
            (false, ShiftOpcode::SRA) => execute_e2_impl::<_, _, false, SraOp>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const IS_IMM: bool,
    OP: ShiftOp,
>(
    pre_compute: &ShiftPreCompute,
    state: &mut VmSegmentState<F, CTX>,
) {
    let rs1 = state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let rs2 = u32::from_le_bytes(rs2);

    // Execute the shift operation
    let rd = <OP as ShiftOp>::compute(rs1, rs2);
    // Write the result back to memory
    state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    state.instret += 1;
    state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_IMM: bool, OP: ShiftOp>(
    pre_compute: &[u8],
    state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &ShiftPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx, const IS_IMM: bool, OP: ShiftOp>(
    pre_compute: &[u8],
    state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftPreCompute> = pre_compute.borrow();
    state.ctx.on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, state);
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftStep<A, NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<(bool, ShiftOpcode)> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let e_u32 = e.as_canonical_u32();
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = ShiftPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        // `d` is always expected to be RV32_REGISTER_AS.
        Ok((is_imm, shift_opcode))
    }
}

trait ShiftOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4];
}
struct SllOp;
struct SrlOp;
struct SraOp;
impl ShiftOp for SllOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 << (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftOp for SrlOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftOp for SraOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}

// Returns (result, limb_shift, bit_shift)
#[inline(always)]
pub(super) fn run_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: ShiftOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], usize, usize) {
    match opcode {
        ShiftOpcode::SLL => run_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        ShiftOpcode::SRL => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, true),
        ShiftOpcode::SRA => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, false),
    }
}

#[inline(always)]
fn run_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], usize, usize) {
    let mut result = [0u8; NUM_LIMBS];

    let (limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);

    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            (((x[i - limb_shift] as u16) << bit_shift)
                | ((x[i - limb_shift - 1] as u16) >> (LIMB_BITS - bit_shift)))
                % (1u16 << LIMB_BITS)
        } else {
            ((x[i - limb_shift] as u16) << bit_shift) % (1u16 << LIMB_BITS)
        } as u8;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn run_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
    logical: bool,
) -> ([u8; NUM_LIMBS], usize, usize) {
    let fill = if logical {
        0
    } else {
        (((1u16 << LIMB_BITS) - 1) as u8) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1))
    };
    let mut result = [fill; NUM_LIMBS];

    let (limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);

    for i in 0..(NUM_LIMBS - limb_shift) {
        let res = if i + limb_shift + 1 < NUM_LIMBS {
            (((x[i + limb_shift] >> bit_shift) as u16)
                | ((x[i + limb_shift + 1] as u16) << (LIMB_BITS - bit_shift)))
                % (1u16 << LIMB_BITS)
        } else {
            (((x[i + limb_shift] >> bit_shift) as u16) | ((fill as u16) << (LIMB_BITS - bit_shift)))
                % (1u16 << LIMB_BITS)
        };
        result[i] = res as u8;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u8]) -> (usize, usize) {
    debug_assert!(NUM_LIMBS * LIMB_BITS <= (1 << LIMB_BITS));
    // We assume `NUM_LIMBS * LIMB_BITS <= 2^LIMB_BITS` so the shift is defined
    // entirely in y[0].
    let shift = (y[0] as usize) % (NUM_LIMBS * LIMB_BITS);
    (shift / LIMB_BITS, shift % LIMB_BITS)
}
