use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, MinimalInstruction, Result,
        StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip, SharedBitwiseOperationLookupChip,
    },
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_big_array::BigArray;
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct LessThanCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub cmp_result: T,

    pub opcode_slt_flag: T,
    pub opcode_sltu_flag: T,

    // Most significant limb of b and c respectively as a field element, will be range
    // checked to be within [-128, 127) if signed, [0, 256) if unsigned.
    pub b_msb_f: T,
    pub c_msb_f: T,

    // 1 at the most significant index i such that b[i] != c[i], otherwise 0. If such
    // an i exists, diff_val = c[i] - b[i] if c[i] > b[i] or b[i] - c[i] else.
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct LessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LessThanCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &LessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_slt_flag, cols.opcode_sltu_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let b = &cols.b;
        let c = &cols.c;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        let c_diff = c[NUM_LIMBS - 1] - cols.c_msb_f;
        builder
            .assert_zero(b_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - b_diff));
        builder
            .assert_zero(c_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - c_diff));

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.c_msb_f - cols.b_msb_f
            } else {
                c[i] - b[i]
            }) * (AB::Expr::from_canonical_u8(2) * cols.cmp_result - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }
        // - If x != y, then prefix_sum = 1 so marker[i] must be 1 iff i is the first index where
        //   diff != 0. Constrains that diff == diff_val where diff_val is non-zero.
        // - If x == y, then prefix_sum = 0 and cmp_result = 0. Here, prefix_sum cannot be 1 because
        //   all diff are zero, making diff == diff_val fails.

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_result);

        // Check if b_msb_f and c_msb_f are in [-128, 127) if signed, [0, 256) if unsigned.
        self.bus
            .send_range(
                cols.b_msb_f
                    + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag,
                cols.c_msb_f
                    + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag,
            )
            .eval(builder, is_valid.clone());

        // Range check to ensure diff_val is non-zero.
        self.bus
            .send_range(cols.diff_val - AB::Expr::ONE, AB::F::ZERO)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(LessThanOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a].into(),
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

pub struct LessThanStep<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> LessThanStep<A, NUM_LIMBS, LIMB_BITS> {
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
    for LessThanStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = ([u8; NUM_LIMBS], [u8; NUM_LIMBS]),
            WriteData = [u8; NUM_LIMBS],
            TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>,
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LessThanOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        debug_assert!(LIMB_BITS <= 8);

        let Instruction { opcode, .. } = instruction;

        let local_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let mut row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let (rs1, rs2) = self.adapter.read(state.memory, instruction, adapter_row);

        let (cmp_result, _, _, _) = run_less_than::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1, &rs2);

        let core_row: &mut LessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        core_row.b = rs1.map(F::from_canonical_u8);
        core_row.c = rs2.map(F::from_canonical_u8);
        core_row.opcode_slt_flag = F::from_bool(local_opcode == LessThanOpcode::SLT);
        core_row.opcode_sltu_flag = F::from_bool(local_opcode == LessThanOpcode::SLTU);

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

        self.adapter
            .write(state.memory, instruction, adapter_row, &output);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter
            .fill_trace_row(mem_helper, self.bitwise_lookup_chip.as_ref(), adapter_row);

        let core_row: &mut LessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let b = core_row.b.map(|x| x.as_canonical_u32() as u8);
        let c = core_row.c.map(|x| x.as_canonical_u32() as u8);
        // It's easier (and faster?) to re-execute
        let local_opcode = if core_row.opcode_slt_flag.is_one() {
            LessThanOpcode::SLT
        } else {
            LessThanOpcode::SLTU
        };
        let (cmp_result, diff_idx, b_sign, c_sign) =
            run_less_than::<NUM_LIMBS, LIMB_BITS>(local_opcode, &b, &c);

        // We range check (b_msb_f + 128) and (c_msb_f + 128) if signed,
        // b_msb_f and c_msb_f if not
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_canonical_u16((1u16 << LIMB_BITS) - b[NUM_LIMBS - 1] as u16),
                b[NUM_LIMBS - 1] - (1u8 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u8(b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1]
                    + (((local_opcode == LessThanOpcode::SLT) as u8) << (LIMB_BITS - 1)),
            )
        };
        let (c_msb_f, c_msb_range) = if c_sign {
            (
                -F::from_canonical_u16((1u16 << LIMB_BITS) - c[NUM_LIMBS - 1] as u16),
                c[NUM_LIMBS - 1] - (1u8 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u8(c[NUM_LIMBS - 1]),
                c[NUM_LIMBS - 1]
                    + (((local_opcode == LessThanOpcode::SLT) as u8) << (LIMB_BITS - 1)),
            )
        };

        let diff_val = if diff_idx == NUM_LIMBS {
            0
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_result {
                c_msb_f - b_msb_f
            } else {
                b_msb_f - c_msb_f
            }
            .as_canonical_u32()
        } else if cmp_result {
            (c[diff_idx] - b[diff_idx]) as u32
        } else {
            (b[diff_idx] - c[diff_idx]) as u32
        };

        self.bitwise_lookup_chip
            .request_range(b_msb_range as u32, c_msb_range as u32);
        if diff_idx != NUM_LIMBS {
            self.bitwise_lookup_chip.request_range(diff_val - 1, 0);
        }

        core_row.diff_val = F::from_canonical_u32(diff_val);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.b_msb_f = b_msb_f;
        core_row.c_msb_f = c_msb_f;
        core_row.diff_val = F::from_canonical_u32(diff_val);
        core_row.diff_marker = array::from_fn(|i| F::from_bool(i == diff_idx));
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> StepExecutorE1<F>
    for LessThanStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<
            F,
            ReadData = ([u8; NUM_LIMBS], [u8; NUM_LIMBS]),
            WriteData = [u8; NUM_LIMBS],
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
        let Instruction { opcode, .. } = instruction;

        let less_than_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (rs1, rs2) = self.adapter.read(state.memory, instruction);

        // Run the comparison
        let (cmp_result, _, _, _) =
            run_less_than::<NUM_LIMBS, LIMB_BITS>(less_than_opcode, &rs1, &rs2);
        let mut rd = [0u8; NUM_LIMBS];
        rd[0] = cmp_result as u8;

        self.adapter.write(state.memory, instruction, &rd);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_less_than<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: LessThanOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && opcode == LessThanOpcode::SLT;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && opcode == LessThanOpcode::SLT;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign, i, x_sign, y_sign);
        }
    }
    (false, NUM_LIMBS, x_sign, y_sign)
}
