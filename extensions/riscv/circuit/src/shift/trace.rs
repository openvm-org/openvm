use std::{array, borrow::BorrowMut, marker::PhantomData};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
    AlignedBytesBorrow,
};
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    air::{ShiftCols, ShiftSraCols},
    core::run_shift,
    ShiftOp,
};

/// Record shared by the three split shift chips. The opcode is fixed per chip, so unlike the
/// combined [`ShiftCoreRecord`](super::ShiftCoreRecord) it does not store `local_opcode`.
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct ShiftSplitRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
}

/// Trace filler for the logical shifts (SLL/SRL), selected by the marker type `OP`.
#[derive(Clone)]
pub struct LogicalShiftFiller<A, OP, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    phantom: PhantomData<OP>,
}

/// Trace filler for the arithmetic right shift (SRA).
#[derive(Clone)]
pub struct ArithShiftFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, OP, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    LogicalShiftFiller<A, OP, NUM_LIMBS, LIMB_BITS>
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
            phantom: PhantomData,
        }
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ArithShiftFiller<A, NUM_LIMBS, LIMB_BITS> {
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

impl<F, A, OP, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for LogicalShiftFiller<A, OP, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
    OP: ShiftOp,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // ShiftCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid ShiftSplitRecord written by the executor
        // during trace generation
        let record: &ShiftSplitRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let (a, limb_shift, bit_shift) =
            run_shift::<NUM_LIMBS, LIMB_BITS>(OP::OPCODE, &record.b, &record.c);

        request_shift_ranges::<NUM_LIMBS, LIMB_BITS>(
            &self.bitwise_lookup_chip,
            &self.range_checker_chip,
            &a,
            &record.b,
            &record.c,
            limb_shift,
            bit_shift,
        );

        let bit_shift_carry = fill_carries::<F, NUM_LIMBS, LIMB_BITS>(
            &self.range_checker_chip,
            OP::IS_LEFT,
            &record.b,
            bit_shift,
        );

        let b = record.b;
        let c = record.c;
        let core_row: &mut ShiftCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        core_row.bit_shift_carry = bit_shift_carry;
        core_row.limb_shift_marker = [F::ZERO; NUM_LIMBS];
        core_row.limb_shift_marker[limb_shift] = F::ONE;
        core_row.bit_shift_marker = [F::ZERO; LIMB_BITS];
        core_row.bit_shift_marker[bit_shift] = F::ONE;
        core_row.bit_multiplier = F::from_usize(1 << bit_shift);

        // `a` aliases the record bytes, so it must be written last.
        core_row.c = c.map(F::from_u8);
        core_row.b = b.map(F::from_u8);
        core_row.a = a.map(F::from_u8);
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for ArithShiftFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // ShiftSraCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid ShiftSplitRecord written by the executor
        // during trace generation
        let record: &ShiftSplitRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let (a, limb_shift, bit_shift) =
            run_shift::<NUM_LIMBS, LIMB_BITS>(ShiftOpcode::SRA, &record.b, &record.c);

        request_shift_ranges::<NUM_LIMBS, LIMB_BITS>(
            &self.bitwise_lookup_chip,
            &self.range_checker_chip,
            &a,
            &record.b,
            &record.c,
            limb_shift,
            bit_shift,
        );

        let bit_shift_carry = fill_carries::<F, NUM_LIMBS, LIMB_BITS>(
            &self.range_checker_chip,
            false,
            &record.b,
            bit_shift,
        );

        let b_sign = record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
        self.bitwise_lookup_chip
            .request_xor(record.b[NUM_LIMBS - 1] as u32, 1 << (LIMB_BITS - 1));

        let b = record.b;
        let c = record.c;
        let core_row: &mut ShiftSraCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        core_row.bit_shift_carry = bit_shift_carry;
        core_row.limb_shift_marker = [F::ZERO; NUM_LIMBS];
        core_row.limb_shift_marker[limb_shift] = F::ONE;
        core_row.bit_shift_marker = [F::ZERO; LIMB_BITS];
        core_row.bit_shift_marker[bit_shift] = F::ONE;
        core_row.b_sign = F::from_u8(b_sign);
        core_row.bit_multiplier = F::from_usize(1 << bit_shift);

        // `a` aliases the record bytes, so it must be written last.
        core_row.c = c.map(F::from_u8);
        core_row.b = b.map(F::from_u8);
        core_row.a = a.map(F::from_u8);
    }
}

/// Requests the byte-range and shift-amount range checks shared by all three chips.
fn request_shift_ranges<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    bitwise_lookup_chip: &SharedBitwiseOperationLookupChip<LIMB_BITS>,
    range_checker_chip: &SharedVariableRangeCheckerChip,
    a: &[u8; NUM_LIMBS],
    b: &[u8; NUM_LIMBS],
    c: &[u8; NUM_LIMBS],
    limb_shift: usize,
    bit_shift: usize,
) {
    for pair in a.chunks_exact(2) {
        bitwise_lookup_chip.request_range(pair[0] as u32, pair[1] as u32);
    }
    for pair in b.chunks_exact(2) {
        bitwise_lookup_chip.request_range(pair[0] as u32, pair[1] as u32);
    }
    for pair in c.chunks_exact(2) {
        bitwise_lookup_chip.request_range(pair[0] as u32, pair[1] as u32);
    }

    let num_bits_log = (NUM_LIMBS * LIMB_BITS).ilog2();
    range_checker_chip.add_count(
        ((c[0] as usize - bit_shift - limb_shift * LIMB_BITS) >> num_bits_log) as u32,
        LIMB_BITS - num_bits_log as usize,
    );
}

/// Computes the per-limb bit-shift carries and requests their range checks.
fn fill_carries<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    range_checker_chip: &SharedVariableRangeCheckerChip,
    is_left: bool,
    b: &[u8; NUM_LIMBS],
    bit_shift: usize,
) -> [F; NUM_LIMBS] {
    if bit_shift == 0 {
        for _ in 0..NUM_LIMBS {
            range_checker_chip.add_count(0, 0);
        }
        [F::ZERO; NUM_LIMBS]
    } else {
        array::from_fn(|i| {
            let carry = if is_left {
                b[i] >> (LIMB_BITS - bit_shift)
            } else {
                b[i] % (1 << bit_shift)
            };
            range_checker_chip.add_count(carry as u32, bit_shift);
            F::from_u8(carry)
        })
    }
}
