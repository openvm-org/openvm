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
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftArithmeticRightCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub is_valid: T,

    // bit_multiplier = 2^bit_shift
    pub bit_multiplier: T,

    // Sign of b for SRA
    pub b_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Part of each b[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; NUM_LIMBS],
}

/// Arithmetic shift-right AIR (SRA).
/// Note: when the shift amount from operand is greater than the number of bits, only shift
/// `shift_amount % num_bits` bits. This matches the RISC-V specs for SRA.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftArithmeticRightCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftArithmeticRightCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftArithmeticRightCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftArithmeticRightCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftArithmeticRightCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftArithmeticRightCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftArithmeticRightCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.is_valid);
        let is_valid: AB::Expr = cols.is_valid.into();

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Constrain that bit_shift, bit_multiplier are correct, i.e. that bit_multiplier =
        // 1 << bit_shift. Because the sum of all bit_shift_marker[i] is constrained to be
        // 1, bit_shift is guaranteed to be in range.
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;

        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            bit_marker_sum += cols.bit_shift_marker[i].into();
            bit_shift += AB::Expr::from_usize(i) * cols.bit_shift_marker[i];

            let mut when_bit_shift = builder.when(cols.bit_shift_marker[i]);
            when_bit_shift.assert_eq(
                cols.bit_multiplier,
                AB::Expr::from_usize(1 << i) * is_valid.clone(),
            );
        }
        builder.assert_eq(bit_marker_sum, is_valid.clone());

        // Check that a is the result of SRA b c.
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for j in 0..NUM_LIMBS {
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift
                        .assert_eq(a[j], cols.b_sign * AB::F::from_usize((1 << LIMB_BITS) - 1));
                } else {
                    let expected_a = if j + i == NUM_LIMBS - 1 {
                        cols.b_sign * (cols.bit_multiplier - AB::F::ONE)
                    } else {
                        cols.bit_shift_carry[j + i + 1].into()
                    } * AB::F::from_usize(1 << LIMB_BITS)
                        + b[j + i]
                        - cols.bit_shift_carry[j + i];
                    when_limb_shift.assert_eq(a[j] * cols.bit_multiplier, expected_a);
                }
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        // Check that bit_shift and limb_shift are correct.
        let num_bits = AB::F::from_usize(NUM_LIMBS * LIMB_BITS);
        self.range_bus
            .range_check(
                (c[0] - limb_shift * AB::F::from_usize(LIMB_BITS) - bit_shift.clone())
                    * num_bits.inverse(),
                LIMB_BITS - ((NUM_LIMBS * LIMB_BITS) as u32).ilog2() as usize,
            )
            .eval(builder, is_valid.clone());

        // Check b_sign & b[NUM_LIMBS - 1] == b_sign using XOR.
        builder.assert_bool(cols.b_sign);

        let mask = AB::F::from_u32(1 << (LIMB_BITS - 1));
        let b_sign_shifted = cols.b_sign * mask;
        self.bitwise_lookup_bus
            .send_xor(
                b[NUM_LIMBS - 1],
                mask,
                b[NUM_LIMBS - 1] + mask - (AB::Expr::from_u32(2) * b_sign_shifted),
            )
            .eval(builder, cols.is_valid);

        for i in 0..(NUM_LIMBS / 2) {
            self.bitwise_lookup_bus
                .send_range(a[i * 2], a[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        // Memory bus checks only packed u16 values; these byte limbs need separate bounds.
        for i in 0..(NUM_LIMBS / 2) {
            self.bitwise_lookup_bus
                .send_range(b[i * 2], b[i * 2 + 1])
                .eval(builder, is_valid.clone());
            self.bitwise_lookup_bus
                .send_range(c[i * 2], c[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        for carry in cols.bit_shift_carry {
            self.range_bus
                .send(carry, bit_shift.clone())
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_u8(ShiftOpcode::SRA as u8),
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
pub struct ShiftArithmeticRightCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy)]
pub struct ShiftArithmeticRightExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone)]
pub struct ShiftArithmeticRightFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftArithmeticRightExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(adapter: A, offset: usize) -> Self {
        assert_eq!(NUM_LIMBS % 2, 0, "Number of limbs must be divisible by 2");
        Self { adapter, offset }
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftArithmeticRightFiller<A, NUM_LIMBS, LIMB_BITS>
{
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

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for ShiftArithmeticRightExecutor<A, NUM_LIMBS, LIMB_BITS>
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
            &'buf mut ShiftArithmeticRightCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_eq!(local_opcode, ShiftOpcode::SRA);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (output, _, _) = run_shift_arithmetic_right::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

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

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for ShiftArithmeticRightFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // ShiftArithmeticRightCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid ShiftArithmeticRightCoreRecord written by the
        // executor during trace generation
        let record: &ShiftArithmeticRightCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let (a, limb_shift, bit_shift) =
            run_shift_arithmetic_right::<NUM_LIMBS, LIMB_BITS>(&record.b, &record.c);

        for pair in a.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        for pair in record.b.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        for pair in record.c.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }

        let num_bits_log = (NUM_LIMBS * LIMB_BITS).ilog2();
        self.range_checker_chip.add_count(
            ((record.c[0] as usize - bit_shift - limb_shift * LIMB_BITS) >> num_bits_log) as u32,
            LIMB_BITS - num_bits_log as usize,
        );

        let bit_shift_carry: [F; NUM_LIMBS] = if bit_shift == 0 {
            for _ in 0..NUM_LIMBS {
                self.range_checker_chip.add_count(0, 0);
            }
            [F::ZERO; NUM_LIMBS]
        } else {
            array::from_fn(|i| {
                let carry = record.b[i] % (1 << bit_shift);
                self.range_checker_chip.add_count(carry as u32, bit_shift);
                F::from_u8(carry)
            })
        };

        let mut limb_shift_marker = [F::ZERO; NUM_LIMBS];
        limb_shift_marker[limb_shift] = F::ONE;
        let mut bit_shift_marker = [F::ZERO; LIMB_BITS];
        bit_shift_marker[bit_shift] = F::ONE;

        let b_sign = F::from_u8(record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1));
        self.bitwise_lookup_chip
            .request_xor(record.b[NUM_LIMBS - 1] as u32, 1 << (LIMB_BITS - 1));

        let core_row: &mut ShiftArithmeticRightCoreCols<F, NUM_LIMBS, LIMB_BITS> =
            core_row.borrow_mut();
        core_row.is_valid = F::ONE;
        core_row.bit_multiplier = F::from_usize(1 << bit_shift);
        core_row.b_sign = b_sign;
        core_row.bit_shift_marker = bit_shift_marker;
        core_row.limb_shift_marker = limb_shift_marker;
        core_row.bit_shift_carry = bit_shift_carry;
        core_row.c = record.c.map(F::from_u8);
        core_row.b = record.b.map(F::from_u8);
        core_row.a = a.map(F::from_u8);
    }
}

// Returns (result, limb_shift, bit_shift)
#[inline(always)]
pub(super) fn run_shift_arithmetic_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], usize, usize) {
    let fill = (((1u16 << LIMB_BITS) - 1) as u8) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1));
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
