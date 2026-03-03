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
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::ShiftWOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct ShiftWCoreCols<T> {
    pub a: [T; RV64_WORD_NUM_LIMBS],
    pub b: [T; RV64_WORD_NUM_LIMBS],
    pub c: [T; RV64_WORD_NUM_LIMBS],

    pub opcode_sll_flag: T,
    pub opcode_srl_flag: T,
    pub opcode_sra_flag: T,

    // Sign bit of the low 32-bit result used to constrain upper-byte sign extension.
    pub result_sign: T,

    // bit_multiplier = 2^bit_shift
    pub bit_multiplier_left: T,
    pub bit_multiplier_right: T,

    // Sign of rs1 for SRAW fill behavior.
    pub b_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; RV64_CELL_BITS],
    pub limb_shift_marker: [T; RV64_WORD_NUM_LIMBS],

    // Part of each x[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; RV64_WORD_NUM_LIMBS],
}

/// RV64 W-shift AIR.
/// Note: W-shifts operate on low 32 bits and use shift amount modulo 32.
#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct ShiftWCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for ShiftWCoreAir {
    fn width(&self) -> usize {
        ShiftWCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for ShiftWCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for ShiftWCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV64_WORD_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &ShiftWCoreCols<_> = local_core.borrow();
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
        builder.assert_bool(cols.result_sign);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;
        let right_shift = cols.opcode_srl_flag + cols.opcode_sra_flag;

        // Constrain that bit_shift, bit_multiplier are correct, i.e. that bit_multiplier =
        // 1 << bit_shift. Because the sum of all bit_shift_marker[i] is constrained to be
        // 1, bit_shift is guaranteed to be in range.
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;

        for i in 0..RV64_CELL_BITS {
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
        // RV64_WORD_NUM_LIMBS * RV64_CELL_BITS.
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..RV64_WORD_NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_canonical_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for j in 0..RV64_WORD_NUM_LIMBS {
                // SLL constraints
                if j < i {
                    when_limb_shift.assert_zero(a[j] * cols.opcode_sll_flag);
                } else {
                    let expected_a_left = if j - i == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j - i - 1].into() * cols.opcode_sll_flag
                    } + b[j - i] * cols.bit_multiplier_left
                        - AB::Expr::from_canonical_usize(1 << RV64_CELL_BITS)
                            * cols.bit_shift_carry[j - i]
                            * cols.opcode_sll_flag;
                    when_limb_shift.assert_eq(a[j] * cols.opcode_sll_flag, expected_a_left);
                }

                // SRL and SRA constraints. Combining with above would require an additional column.
                if j + i > RV64_WORD_NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        a[j] * right_shift.clone(),
                        cols.b_sign * AB::F::from_canonical_usize((1 << RV64_CELL_BITS) - 1),
                    );
                } else {
                    let expected_a_right = if j + i == RV64_WORD_NUM_LIMBS - 1 {
                        cols.b_sign * (cols.bit_multiplier_right - AB::F::ONE)
                    } else {
                        cols.bit_shift_carry[j + i + 1].into() * right_shift.clone()
                    } * AB::F::from_canonical_usize(1 << RV64_CELL_BITS)
                        + right_shift.clone() * (b[j + i] - cols.bit_shift_carry[j + i]);
                    when_limb_shift.assert_eq(a[j] * cols.bit_multiplier_right, expected_a_right);
                }
            }
        }
        builder.when(is_valid.clone()).assert_one(limb_marker_sum);

        // Check that bit_shift and limb_shift are correct.
        let num_bits = AB::F::from_canonical_usize(RV64_WORD_NUM_LIMBS * RV64_CELL_BITS);
        self.range_bus
            .range_check(
                (c[0]
                    - limb_shift * AB::F::from_canonical_usize(RV64_CELL_BITS)
                    - bit_shift.clone())
                    * num_bits.inverse(),
                RV64_CELL_BITS - ((RV64_WORD_NUM_LIMBS * RV64_CELL_BITS) as u32).ilog2() as usize,
            )
            .eval(builder, is_valid.clone());

        // Check b_sign & b[word_msl] == b_sign using XOR.
        builder.assert_bool(cols.b_sign);
        builder
            .when(not(cols.opcode_sra_flag))
            .assert_zero(cols.b_sign);

        let mask = AB::F::from_canonical_u32(1 << (RV64_CELL_BITS - 1));
        let b_sign_shifted = cols.b_sign * mask;
        let result_sign_shifted = cols.result_sign * mask;
        self.bitwise_lookup_bus
            .send_xor(
                b[RV64_WORD_NUM_LIMBS - 1],
                mask,
                b[RV64_WORD_NUM_LIMBS - 1] + mask
                    - (AB::Expr::from_canonical_u32(2) * b_sign_shifted),
            )
            .eval(builder, cols.opcode_sra_flag);

        // result_sign must match output sign bit.
        self.bitwise_lookup_bus
            .send_xor(
                a[RV64_WORD_NUM_LIMBS - 1],
                mask,
                a[RV64_WORD_NUM_LIMBS - 1] + mask
                    - (AB::Expr::from_canonical_u32(2) * result_sign_shifted),
            )
            .eval(builder, is_valid.clone());

        let sign_extend_limb =
            AB::Expr::from_canonical_u32((1 << RV64_CELL_BITS) - 1) * cols.result_sign;
        let write_data: [AB::Expr; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|i| {
            if i < RV64_WORD_NUM_LIMBS {
                a[i].into()
            } else {
                sign_extend_limb.clone()
            }
        });

        for i in 0..(RV64_WORD_NUM_LIMBS / 2) {
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
                .zip(ShiftWOpcode::iter())
                .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
                }),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [write_data].into(),
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
pub struct ShiftWCoreRecord {
    pub b: [u8; RV64_WORD_NUM_LIMBS],
    pub c: [u8; RV64_WORD_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy)]
pub struct ShiftWExecutor<A> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone)]
pub struct ShiftWFiller<A> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A> ShiftWExecutor<A> {
    pub fn new(adapter: A, offset: usize) -> Self {
        assert_eq!(
            RV64_WORD_NUM_LIMBS % 2,
            0,
            "Number of limbs must be divisible by 2"
        );
        Self { adapter, offset }
    }
}

impl<A> ShiftWFiller<A> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
        offset: usize,
    ) -> Self {
        assert_eq!(
            RV64_WORD_NUM_LIMBS % 2,
            0,
            "Number of limbs must be divisible by 2"
        );
        Self {
            adapter,
            offset,
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for ShiftWExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; RV64_WORD_NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; RV64_REGISTER_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut ShiftWCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftWOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = ShiftWOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (output, _, _) = run_shift_w(local_opcode, &rs1, &rs2);

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

impl<F, A> TraceFiller<F> for ShiftWFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // ShiftWCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid ShiftWCoreRecord written by the executor
        // during trace generation
        let record: &ShiftWCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut ShiftWCoreCols<F> = core_row.borrow_mut();

        let opcode = ShiftWOpcode::from_usize(record.local_opcode as usize);
        let (a, limb_shift, bit_shift) = run_shift_w(opcode, &record.b, &record.c);

        for pair in a[..RV64_WORD_NUM_LIMBS].chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }

        let num_bits_log = (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS).ilog2();
        self.range_checker_chip.add_count(
            ((record.c[0] as usize - bit_shift - limb_shift * RV64_CELL_BITS) >> num_bits_log)
                as u32,
            RV64_CELL_BITS - num_bits_log as usize,
        );

        core_row.bit_shift_carry = if bit_shift == 0 {
            for _ in 0..RV64_WORD_NUM_LIMBS {
                self.range_checker_chip.add_count(0, 0);
            }
            [F::ZERO; RV64_WORD_NUM_LIMBS]
        } else {
            array::from_fn(|i| {
                let carry = match opcode {
                    ShiftWOpcode::SLLW => record.b[i] >> (RV64_CELL_BITS - bit_shift),
                    _ => record.b[i] % (1 << bit_shift),
                };
                self.range_checker_chip.add_count(carry as u32, bit_shift);
                F::from_canonical_u8(carry)
            })
        };

        core_row.limb_shift_marker = [F::ZERO; RV64_WORD_NUM_LIMBS];
        core_row.limb_shift_marker[limb_shift] = F::ONE;
        core_row.bit_shift_marker = [F::ZERO; RV64_CELL_BITS];
        core_row.bit_shift_marker[bit_shift] = F::ONE;

        core_row.result_sign =
            F::from_canonical_u8(a[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS as u8 - 1));
        self.bitwise_lookup_chip
            .request_xor(a[RV64_WORD_NUM_LIMBS - 1] as u32, 1 << (RV64_CELL_BITS - 1));
        core_row.b_sign = F::ZERO;
        if opcode == ShiftWOpcode::SRAW {
            self.bitwise_lookup_chip.request_xor(
                record.b[RV64_WORD_NUM_LIMBS - 1] as u32,
                1 << (RV64_CELL_BITS - 1),
            );
            core_row.b_sign =
                F::from_canonical_u8(record.b[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1));
        }

        core_row.bit_multiplier_right = match opcode {
            ShiftWOpcode::SLLW => F::ZERO,
            _ => F::from_canonical_usize(1 << bit_shift),
        };
        core_row.bit_multiplier_left = match opcode {
            ShiftWOpcode::SLLW => F::from_canonical_usize(1 << bit_shift),
            _ => F::ZERO,
        };

        core_row.opcode_sra_flag = F::from_bool(opcode == ShiftWOpcode::SRAW);
        core_row.opcode_srl_flag = F::from_bool(opcode == ShiftWOpcode::SRLW);
        core_row.opcode_sll_flag = F::from_bool(opcode == ShiftWOpcode::SLLW);

        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = array::from_fn(|i| F::from_canonical_u8(a[i]));
    }
}

// Returns (result, limb_shift, bit_shift)
#[inline(always)]
pub(super) fn run_shift_w(
    opcode: ShiftWOpcode,
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
) -> ([u8; RV64_REGISTER_NUM_LIMBS], usize, usize) {
    let (word_result, limb_shift, bit_shift) = match opcode {
        ShiftWOpcode::SLLW => run_shift_left_w(x, y),
        ShiftWOpcode::SRLW => run_shift_right_w(x, y, true),
        ShiftWOpcode::SRAW => run_shift_right_w(x, y, false),
    };
    let sign_extend_limb = ((1u16 << RV64_CELL_BITS) - 1) as u8
        * (word_result[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS as u8 - 1));
    let mut result = [sign_extend_limb; RV64_REGISTER_NUM_LIMBS];
    result[..RV64_WORD_NUM_LIMBS].copy_from_slice(&word_result);
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn run_shift_left_w(
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
) -> ([u8; RV64_WORD_NUM_LIMBS], usize, usize) {
    let mut result = [0u8; RV64_WORD_NUM_LIMBS];
    let (limb_shift, bit_shift) = get_shift_w(y);

    for i in limb_shift..RV64_WORD_NUM_LIMBS {
        result[i] = if i > limb_shift {
            (((x[i - limb_shift] as u16) << bit_shift)
                | ((x[i - limb_shift - 1] as u16) >> (RV64_CELL_BITS - bit_shift)))
                % (1u16 << RV64_CELL_BITS)
        } else {
            ((x[i - limb_shift] as u16) << bit_shift) % (1u16 << RV64_CELL_BITS)
        } as u8;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn run_shift_right_w(
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
    logical: bool,
) -> ([u8; RV64_WORD_NUM_LIMBS], usize, usize) {
    let fill = if logical {
        0
    } else {
        (((1u16 << RV64_CELL_BITS) - 1) as u8)
            * (x[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS as u8 - 1))
    };
    let mut result = [fill; RV64_WORD_NUM_LIMBS];
    let (limb_shift, bit_shift) = get_shift_w(y);

    for i in 0..(RV64_WORD_NUM_LIMBS - limb_shift) {
        let res = if i + limb_shift + 1 < RV64_WORD_NUM_LIMBS {
            (((x[i + limb_shift] >> bit_shift) as u16)
                | ((x[i + limb_shift + 1] as u16) << (RV64_CELL_BITS - bit_shift)))
                % (1u16 << RV64_CELL_BITS)
        } else {
            (((x[i + limb_shift] >> bit_shift) as u16)
                | ((fill as u16) << (RV64_CELL_BITS - bit_shift)))
                % (1u16 << RV64_CELL_BITS)
        };
        result[i] = res as u8;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn get_shift_w(y: &[u8; RV64_WORD_NUM_LIMBS]) -> (usize, usize) {
    // For W-shifts, the shift amount is determined by the low 32 bits, and in this byte-limb
    // representation that is fully determined by y[0].
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS);
    (shift / RV64_CELL_BITS, shift % RV64_CELL_BITS)
}
