use std::{
    array,
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
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchLessThanCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    // Boolean result of a op b. Should branch if and only if cmp_result = 1.
    pub cmp_result: T,
    pub imm: T,

    pub opcode_blt_flag: T,
    pub opcode_bltu_flag: T,
    pub opcode_bge_flag: T,
    pub opcode_bgeu_flag: T,

    // Most significant limb of a and b respectively as a field element, will be range
    // checked to be within [-128, 127) if signed and [0, 256) if unsigned.
    pub a_msb_f: T,
    pub b_msb_f: T,

    // 1 if a < b, 0 otherwise.
    pub cmp_lt: T,

    // 1 at the most significant index i such that a[i] != b[i], otherwise 0. If such
    // an i exists, diff_val = b[i] - a[i].
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BranchLessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BranchLessThanCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: Default,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BranchLessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_blt_flag,
            cols.opcode_bltu_flag,
            cols.opcode_bge_flag,
            cols.opcode_bgeu_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let lt = cols.opcode_blt_flag + cols.opcode_bltu_flag;
        let ge = cols.opcode_bge_flag + cols.opcode_bgeu_flag;
        let signed = cols.opcode_blt_flag + cols.opcode_bge_flag;
        builder.assert_eq(
            cols.cmp_lt,
            cols.cmp_result * lt.clone() + not(cols.cmp_result) * ge.clone(),
        );

        let a = &cols.a;
        let b = &cols.b;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        // Check if a_msb_f and b_msb_f are signed values of a[NUM_LIMBS - 1] and b[NUM_LIMBS - 1]
        // in prime field F.
        let a_diff = a[NUM_LIMBS - 1] - cols.a_msb_f;
        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        builder
            .assert_zero(a_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - a_diff));
        builder
            .assert_zero(b_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - b_diff));

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.b_msb_f - cols.a_msb_f
            } else {
                b[i] - a[i]
            }) * (AB::Expr::from_canonical_u8(2) * cols.cmp_lt - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }
        // - If x != y, then prefix_sum = 1 so marker[i] must be 1 iff i is the first index where
        //   diff != 0. Constrains that diff == diff_val where diff_val is non-zero.
        // - If x == y, then prefix_sum = 0 and cmp_lt = 0. Here, prefix_sum cannot be 1 because all
        //   diff are zero, making diff == diff_val fails.

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_lt);

        // Check if a_msb_f and b_msb_f are in [-128, 127) if signed, [0, 256) if unsigned.
        self.bus
            .send_range(
                cols.a_msb_f + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * signed.clone(),
                cols.b_msb_f + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * signed.clone(),
            )
            .eval(builder, is_valid.clone());

        // Range check to ensure diff_val is non-zero.
        self.bus
            .send_range(cols.diff_val - AB::Expr::ONE, AB::F::ZERO)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(BranchLessThanOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        let to_pc = from_pc
            + cols.cmp_result * cols.imm
            + not(cols.cmp_result) * AB::Expr::from_canonical_u32(DEFAULT_PC_STEP);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
            writes: Default::default(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: cols.imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

pub struct BranchLessThanStep<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BranchLessThanStep<A, NUM_LIMBS, LIMB_BITS>
{
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
    for BranchLessThanStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData = (),
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BranchLessThanOpcode::from_usize(opcode - self.offset)
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
        let &Instruction { opcode, c: imm, .. } = instruction;

        let blt_opcode = BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, adapter_row)
            .into();

        let (cmp_result, diff_idx, a_sign, b_sign) =
            run_cmp::<NUM_LIMBS, LIMB_BITS>(blt_opcode, &rs1, &rs2);

        let signed = matches!(
            blt_opcode,
            BranchLessThanOpcode::BLT | BranchLessThanOpcode::BGE
        );
        let ge_opcode = matches!(
            blt_opcode,
            BranchLessThanOpcode::BGE | BranchLessThanOpcode::BGEU
        );
        let cmp_lt = cmp_result ^ ge_opcode;

        let a = rs1.map(u32::from);
        let b = rs2.map(u32::from);

        // We range check (a_msb_f + 128) and (b_msb_f + 128) if signed,
        // a_msb_f and b_msb_f if not
        let (a_msb_f, a_msb_range) = if a_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - a[NUM_LIMBS - 1]),
                a[NUM_LIMBS - 1] - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(a[NUM_LIMBS - 1]),
                a[NUM_LIMBS - 1] + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1] - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1] + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };

        let diff_val = if diff_idx == NUM_LIMBS {
            0
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_lt {
                b_msb_f - a_msb_f
            } else {
                a_msb_f - b_msb_f
            }
            .as_canonical_u32()
        } else if cmp_lt {
            b[diff_idx] - a[diff_idx]
        } else {
            a[diff_idx] - b[diff_idx]
        };

        let core_row: &mut BranchLessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        core_row.a = rs1.map(F::from_canonical_u8);
        core_row.b = rs2.map(F::from_canonical_u8);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.cmp_lt = F::from_bool(cmp_lt);
        core_row.imm = imm;
        core_row.a_msb_f = a_msb_f;
        core_row.b_msb_f = b_msb_f;
        core_row.diff_marker = array::from_fn(|i| F::from_bool(i == diff_idx));
        core_row.diff_val = F::from_canonical_u32(diff_val);
        core_row.opcode_blt_flag = F::from_bool(blt_opcode == BranchLessThanOpcode::BLT);
        core_row.opcode_bltu_flag = F::from_bool(blt_opcode == BranchLessThanOpcode::BLTU);
        core_row.opcode_bge_flag = F::from_bool(blt_opcode == BranchLessThanOpcode::BGE);
        core_row.opcode_bgeu_flag = F::from_bool(blt_opcode == BranchLessThanOpcode::BGEU);

        if cmp_result {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        // TODO(ayush): move to fill_trace_row
        self.bitwise_lookup_chip
            .request_range(a_msb_range, b_msb_range);

        if diff_idx != NUM_LIMBS {
            self.bitwise_lookup_chip.request_range(diff_val - 1, 0);
        }

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, _core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> StepExecutorE1<F>
    for BranchLessThanStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterExecutorE1<F, ReadData: Into<[[u8; NUM_LIMBS]; 2]>, WriteData = ()>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let blt_opcode = BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [rs1, rs2] = self.adapter.read(state, instruction).into();

        // TODO(ayush): probably don't need the other values
        let (cmp_result, _, _, _) = run_cmp::<NUM_LIMBS, LIMB_BITS>(blt_opcode, &rs1, &rs2);

        if cmp_result {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_cmp<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    local_opcode: BranchLessThanOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let signed =
        local_opcode == BranchLessThanOpcode::BLT || local_opcode == BranchLessThanOpcode::BGE;
    let ge_op =
        local_opcode == BranchLessThanOpcode::BGE || local_opcode == BranchLessThanOpcode::BGEU;
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign ^ ge_op, i, x_sign, y_sign);
        }
    }
    (ge_op, NUM_LIMBS, x_sign, y_sign)
}
