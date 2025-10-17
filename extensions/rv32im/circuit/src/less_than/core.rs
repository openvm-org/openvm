use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
    AlignedBytesBorrow,
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
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LessThanCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LessThanExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct LessThanFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut LessThanCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LessThanOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        debug_assert!(LIMB_BITS <= 8);
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        let (cmp_result, _, _, _) = run_less_than::<NUM_LIMBS, LIMB_BITS>(
            core_record.local_opcode == LessThanOpcode::SLT as u8,
            &rs1,
            &rs2,
        );

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

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

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for LessThanFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // LessThanCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid LessThanCoreRecord written by the executor
        // during trace generation
        let record: &LessThanCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut LessThanCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let is_slt = record.local_opcode == LessThanOpcode::SLT as u8;
        let (cmp_result, diff_idx, b_sign, c_sign) =
            run_less_than::<NUM_LIMBS, LIMB_BITS>(is_slt, &record.b, &record.c);

        // We range check (b_msb_f + 128) and (c_msb_f + 128) if signed,
        // b_msb_f and c_msb_f if not
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_canonical_u16((1u16 << LIMB_BITS) - record.b[NUM_LIMBS - 1] as u16),
                record.b[NUM_LIMBS - 1] - (1u8 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u8(record.b[NUM_LIMBS - 1]),
                record.b[NUM_LIMBS - 1] + ((is_slt as u8) << (LIMB_BITS - 1)),
            )
        };
        let (c_msb_f, c_msb_range) = if c_sign {
            (
                -F::from_canonical_u16((1u16 << LIMB_BITS) - record.c[NUM_LIMBS - 1] as u16),
                record.c[NUM_LIMBS - 1] - (1u8 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u8(record.c[NUM_LIMBS - 1]),
                record.c[NUM_LIMBS - 1] + ((is_slt as u8) << (LIMB_BITS - 1)),
            )
        };

        core_row.diff_val = if diff_idx == NUM_LIMBS {
            F::ZERO
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_result {
                c_msb_f - b_msb_f
            } else {
                b_msb_f - c_msb_f
            }
        } else if cmp_result {
            F::from_canonical_u8(record.c[diff_idx] - record.b[diff_idx])
        } else {
            F::from_canonical_u8(record.b[diff_idx] - record.c[diff_idx])
        };

        self.bitwise_lookup_chip
            .request_range(b_msb_range as u32, c_msb_range as u32);

        core_row.diff_marker = [F::ZERO; NUM_LIMBS];
        if diff_idx != NUM_LIMBS {
            self.bitwise_lookup_chip
                .request_range(core_row.diff_val.as_canonical_u32() - 1, 0);
            core_row.diff_marker[diff_idx] = F::ONE;
        }

        core_row.c_msb_f = c_msb_f;
        core_row.b_msb_f = b_msb_f;
        core_row.opcode_sltu_flag = F::from_bool(!is_slt);
        core_row.opcode_slt_flag = F::from_bool(is_slt);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_less_than<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    is_slt: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign, i, x_sign, y_sign);
        }
    }
    (false, NUM_LIMBS, x_sign, y_sign)
}
