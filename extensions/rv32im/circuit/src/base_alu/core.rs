use std::{
    array,
    borrow::{Borrow, BorrowMut},
    iter::zip,
};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterTraceStep, ExecuteFunc,
        MinimalInstruction, PreComputeInstruction, Result, StepExecutorE1, TraceStep,
        VmAdapterInterface, VmCoreAir, VmSegmentState, VmStateMut,
    },
    next_instruction,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::imm_to_bytes;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BaseAluCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BaseAluCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BaseAluCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &BaseAluCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_add_flag,
            cols.opcode_sub_flag,
            cols.opcode_xor_flag,
            cols.opcode_or_flag,
            cols.opcode_and_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // For ADD, define carry[i] = (b[i] + c[i] + carry[i - 1] - a[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= a[i] < 2^LIMB_BITS, it can be proven that
        // a[i] = (b[i] + c[i]) % 2^LIMB_BITS as necessary. The same holds for SUB when
        // carry[i] is (a[i] + c[i] - b[i] + carry[i - 1]) / 2^LIMB_BITS.
        let mut carry_add: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let mut carry_sub: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            // We explicitly separate the constraints for ADD and SUB in order to keep degree
            // cubic. Because we constrain that the carry (which is arbitrary) is bool, if
            // carry has degree larger than 1 the max-degree constrain could be at least 4.
            carry_add[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_add_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (a[i] + c[i] - b[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_sub_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // Interaction with BitwiseOperationLookup to range check a for ADD and SUB, and
        // constrain a's correctness for XOR, OR, and AND.
        let bitwise = cols.opcode_xor_flag + cols.opcode_or_flag + cols.opcode_and_flag;
        for i in 0..NUM_LIMBS {
            let x = not::<AB::Expr>(bitwise.clone()) * a[i] + bitwise.clone() * b[i];
            let y = not::<AB::Expr>(bitwise.clone()) * a[i] + bitwise.clone() * c[i];
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_or_flag * ((AB::Expr::from_canonical_u32(2) * a[i]) - b[i] - c[i])
                + cols.opcode_and_flag * (b[i] + c[i] - (AB::Expr::from_canonical_u32(2) * a[i]));
            self.bus
                .send_xor(x, y, x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(BaseAluOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
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

pub struct BaseAluStep<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAluStep<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        Self {
            adapter,
            offset,
            bitwise_lookup_chip,
        }
    }
}

impl<F, CTX, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceStep<F, CTX>
    for BaseAluStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, adapter_row)
            .into();

        let rd = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1, &rs2);

        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        core_row.a = rd.map(F::from_canonical_u8);
        core_row.b = rs1.map(F::from_canonical_u8);
        core_row.c = rs2.map(F::from_canonical_u8);
        core_row.opcode_add_flag = F::from_bool(local_opcode == BaseAluOpcode::ADD);
        core_row.opcode_sub_flag = F::from_bool(local_opcode == BaseAluOpcode::SUB);
        core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluOpcode::XOR);
        core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluOpcode::OR);
        core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluOpcode::AND);

        self.adapter
            .write(state.memory, instruction, adapter_row, &[rd].into());

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        if core_row.opcode_add_flag == F::ONE || core_row.opcode_sub_flag == F::ONE {
            for a_val in core_row.a.map(|x| x.as_canonical_u32()) {
                self.bitwise_lookup_chip.request_xor(a_val, a_val);
            }
        } else {
            let b = core_row.b.map(|x| x.as_canonical_u32());
            let c = core_row.c.map(|x| x.as_canonical_u32());
            for (b_val, c_val) in zip(b, c) {
                self.bitwise_lookup_chip.request_xor(b_val, c_val);
            }
        }
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BaseAluPreCompute {
    opcode: BaseAluOpcode,
    c: u32,
    a: u8,
    b: u8,
    is_imm: bool,
}

impl<F, A, const LIMB_BITS: usize> StepExecutorE1<F>
    for BaseAluStep<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn execute_e1<Ctx>(&self) -> ExecuteFunc<F, Ctx>
    where
        Ctx: E1E2ExecutionCtx,
    {
        execute_e1_impl
    }

    // fn execute_metered(
    //     &self,
    //     state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
    //     instruction: &Instruction<F>,
    //     chip_index: usize,
    // ) -> Result<()> {
    //     self.execute_e1(state, instruction)?;
    //     state.ctx.trace_heights[chip_index] += 1;
    //
    //     Ok(())
    // }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
    }

    #[inline(always)]
    fn pre_compute(&self, inst: &Instruction<F>, data: &mut [u8]) {
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        let e_u32 = e.as_canonical_u32();
        assert!(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS);
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = BaseAluPreCompute {
            opcode: BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset)),
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            is_imm,
        };
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) -> Result<()> {
    let next_inst = inst.offset(1);
    let inst = &*inst;
    let pre_compute: &BaseAluPreCompute = inst.pre_compute.borrow();
    let rs1 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if pre_compute.is_imm {
        pre_compute.c.to_le_bytes()
    } else {
        vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };

    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = match pre_compute.opcode {
        BaseAluOpcode::ADD => rs1.wrapping_add(rs2),
        BaseAluOpcode::SUB => rs1.wrapping_sub(rs2),
        BaseAluOpcode::XOR => rs1 ^ rs2,
        BaseAluOpcode::OR => rs1 | rs2,
        BaseAluOpcode::AND => rs1 & rs2,
    };
    let rd = rd.to_le_bytes();
    vm_state.vm_write::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
    next_instruction!(next_inst, vm_state)
}

#[inline(always)]
pub(super) fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 8, "specialize for bytes");
    match opcode {
        BaseAluOpcode::ADD => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::SUB => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::XOR => run_xor::<NUM_LIMBS>(x, y),
        BaseAluOpcode::OR => run_or::<NUM_LIMBS>(x, y),
        BaseAluOpcode::AND => run_and::<NUM_LIMBS>(x, y),
    }
}

#[inline(always)]
fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut overflow =
            (x[i] as u16) + (y[i] as u16) + if i > 0 { carry[i - 1] as u16 } else { 0 };
        carry[i] = (overflow >> LIMB_BITS) as u8;
        overflow &= (1u16 << LIMB_BITS) - 1;
        z[i] = overflow as u8;
    }
    z
}

#[inline(always)]
fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] as u16 + if i > 0 { carry[i - 1] as u16 } else { 0 };
        if x[i] as u16 >= rhs {
            z[i] = x[i] - rhs as u8;
            carry[i] = 0;
        } else {
            z[i] = (x[i] as u16 + (1u16 << LIMB_BITS) - rhs) as u8;
            carry[i] = 1;
        }
    }
    z
}

#[inline(always)]
fn run_xor<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

#[inline(always)]
fn run_or<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

#[inline(always)]
fn run_and<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}
