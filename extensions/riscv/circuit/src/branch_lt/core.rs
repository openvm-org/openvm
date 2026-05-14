use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
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
    // checked to be within [-2^(LIMB_BITS-1), 2^(LIMB_BITS-1)) if signed and [0, 2^LIMB_BITS) if
    // unsigned.
    pub a_msb_f: T,
    pub b_msb_f: T,

    // 1 if a < b, 0 otherwise.
    pub cmp_lt: T,

    // 1 at the most significant index i such that a[i] != b[i], otherwise 0. If such
    // an i exists, diff_val = b[i] - a[i].
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BranchLessThanCoreCols<u16, NUM_LIMBS, LIMB_BITS>)]
pub struct BranchLessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
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
        builder.assert_zero(a_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - a_diff));
        builder.assert_zero(b_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - b_diff));

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.b_msb_f - cols.a_msb_f
            } else {
                b[i] - a[i]
            }) * (AB::Expr::from_u8(2) * cols.cmp_lt - AB::Expr::ONE);
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

        // Check that a_msb_f and b_msb_f are in [-2^(LIMB_BITS-1), 2^(LIMB_BITS-1)) if signed,
        // [0, 2^LIMB_BITS) if unsigned. Shift by 2^(LIMB_BITS-1) when signed so the value lands
        // in [0, 2^LIMB_BITS) for a uniform range check.
        let sign_shift = AB::Expr::from_u32(1 << (LIMB_BITS - 1)) * signed.clone();
        self.range_bus
            .range_check(cols.a_msb_f + sign_shift.clone(), LIMB_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift, LIMB_BITS)
            .eval(builder, is_valid.clone());

        // Range-check the non-MSB read limbs a[0..NUM_LIMBS-1] and b[0..NUM_LIMBS-1] to
        // [0, 2^LIMB_BITS) so each limb has a canonical decomposition matching the packed
        // memory-bus field element.
        for i in 0..NUM_LIMBS - 1 {
            self.range_bus
                .range_check(a[i], LIMB_BITS)
                .eval(builder, is_valid.clone());
            self.range_bus
                .range_check(b[i], LIMB_BITS)
                .eval(builder, is_valid.clone());
        }

        // Range check to ensure diff_val is non-zero (`diff_val - 1` lies in [0, 2^LIMB_BITS),
        // so diff_val is in [1, 2^LIMB_BITS + 1)).
        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, LIMB_BITS)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(BranchLessThanOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(opcode as u8)
            })
            + AB::Expr::from_usize(self.offset);

        let to_pc = from_pc
            + cols.cmp_result * cols.imm
            + not(cols.cmp_result) * AB::Expr::from_u32(DEFAULT_PC_STEP);

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BranchLessThanCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [u16; NUM_LIMBS],
    pub b: [u16; NUM_LIMBS],
    pub imm: u32,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BranchLessThanExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct BranchLessThanFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData: Into<[[u16; NUM_LIMBS]; 2]>, WriteData = ()>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BranchLessThanCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BranchLessThanOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.a = rs1;
        core_record.b = rs2;
        core_record.imm = imm.as_canonical_u32();
        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        if run_cmp::<NUM_LIMBS, LIMB_BITS>(core_record.local_opcode, &rs1, &rs2).0 {
            *state.pc = (F::from_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for BranchLessThanFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // BranchLessThanCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        // SAFETY: core_row contains a valid BranchLessThanCoreRecord written by the executor
        // during trace generation
        let record: &BranchLessThanCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let core_row: &mut BranchLessThanCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let signed = record.local_opcode == BranchLessThanOpcode::BLT as u8
            || record.local_opcode == BranchLessThanOpcode::BGE as u8;
        let ge_op = record.local_opcode == BranchLessThanOpcode::BGE as u8
            || record.local_opcode == BranchLessThanOpcode::BGEU as u8;

        let (cmp_result, diff_idx, a_sign, b_sign) =
            run_cmp::<NUM_LIMBS, LIMB_BITS>(record.local_opcode, &record.a, &record.b);

        let cmp_lt = cmp_result ^ ge_op;

        // We range check (a_msb_f + 2^(LIMB_BITS-1)) and (b_msb_f + 2^(LIMB_BITS-1)) if signed,
        // a_msb_f and b_msb_f if not
        let (a_msb_f, a_msb_range) = if a_sign {
            (
                -F::from_u32((1 << LIMB_BITS) - record.a[NUM_LIMBS - 1] as u32),
                record.a[NUM_LIMBS - 1] as u32 - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_u32(record.a[NUM_LIMBS - 1] as u32),
                record.a[NUM_LIMBS - 1] as u32 + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_u32((1 << LIMB_BITS) - record.b[NUM_LIMBS - 1] as u32),
                record.b[NUM_LIMBS - 1] as u32 - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_u32(record.b[NUM_LIMBS - 1] as u32),
                record.b[NUM_LIMBS - 1] as u32 + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };

        core_row.diff_val = if diff_idx == NUM_LIMBS {
            F::ZERO
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_lt {
                b_msb_f - a_msb_f
            } else {
                a_msb_f - b_msb_f
            }
        } else if cmp_lt {
            F::from_u32(record.b[diff_idx] as u32 - record.a[diff_idx] as u32)
        } else {
            F::from_u32(record.a[diff_idx] as u32 - record.b[diff_idx] as u32)
        };

        self.range_checker_chip.add_count(a_msb_range, LIMB_BITS);
        self.range_checker_chip.add_count(b_msb_range, LIMB_BITS);

        // Mirror AIR's non-MSB per-limb range-checks for a[i] and b[i].
        for i in 0..NUM_LIMBS - 1 {
            self.range_checker_chip
                .add_count(record.a[i] as u32, LIMB_BITS);
            self.range_checker_chip
                .add_count(record.b[i] as u32, LIMB_BITS);
        }

        core_row.diff_marker = [F::ZERO; NUM_LIMBS];

        if diff_idx != NUM_LIMBS {
            self.range_checker_chip
                .add_count(core_row.diff_val.as_canonical_u32() - 1, LIMB_BITS);
            core_row.diff_marker[diff_idx] = F::ONE;
        }

        core_row.cmp_lt = F::from_bool(cmp_lt);
        core_row.b_msb_f = b_msb_f;
        core_row.a_msb_f = a_msb_f;
        core_row.opcode_bgeu_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BGEU as u8);
        core_row.opcode_bge_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BGE as u8);
        core_row.opcode_bltu_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BLTU as u8);
        core_row.opcode_blt_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BLT as u8);

        core_row.imm = F::from_u32(record.imm);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.b = record.b.map(|v| F::from_u32(v as u32));
        core_row.a = record.a.map(|v| F::from_u32(v as u32));
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_cmp<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    local_opcode: u8,
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let signed = local_opcode == BranchLessThanOpcode::BLT as u8
        || local_opcode == BranchLessThanOpcode::BGE as u8;
    let ge_op = local_opcode == BranchLessThanOpcode::BGE as u8
        || local_opcode == BranchLessThanOpcode::BGEU as u8;
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign ^ ge_op, i, x_sign, y_sign);
        }
    }
    (ge_op, NUM_LIMBS, x_sign, y_sign)
}
