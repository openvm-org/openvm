use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::{
    core::{is_rotate_imm_opcode, BITMANIP_OFFSET},
    rotate::{fill_rotate_aux, run_rotate},
    BITMANIP_LIMB_BITS,
};

/// Rotate-right-by-immediate (RORI at 4 limbs, RORIW at 2 limbs behind the
/// sign-extending W adapter).
///
/// Reuses the left-rotate recombination of [`super::rotate`]: the markers
/// encode the normalized left-rotate amount `idx = (width - shamt) % width`,
/// and the instruction immediate is bound to `shamt` via the expression
/// `width * (is_valid - is_idx_zero) - idx`, where `is_idx_zero` is the
/// product of the two zero markers (so `shamt = 0` maps to `idx = 0`, not
/// `width`).
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipRotateImmCoreCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub bit_shift_marker: [T; BITMANIP_LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipRotateImmCoreCols<u8, NUM_LIMBS>)]
pub struct BitManipRotateImmCoreAir<const NUM_LIMBS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    /// Local opcode of the rotate-right-immediate this instantiation serves.
    pub rori_opcode: usize,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for BitManipRotateImmCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        BitManipRotateImmCoreCols::<F, NUM_LIMBS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F>
    for BitManipRotateImmCoreAir<NUM_LIMBS>
{
}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for BitManipRotateImmCoreAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipRotateImmCoreCols<_, NUM_LIMBS> = local_core.borrow();
        let width_bits = NUM_LIMBS * BITMANIP_LIMB_BITS;

        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;
        for bit in 0..BITMANIP_LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[bit]);
            let marker: AB::Expr = cols.bit_shift_marker[bit].into();
            bit_marker_sum += marker.clone();
            bit_shift += AB::Expr::from_usize(bit) * marker.clone();
            bit_multiplier += AB::Expr::from_usize(1 << bit) * marker.clone();
            carry_multiplier += AB::Expr::from_usize(1 << (BITMANIP_LIMB_BITS - bit)) * marker;
        }
        builder.assert_bool(bit_marker_sum.clone());
        let is_valid = bit_marker_sum;

        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for limb in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[limb]);
            limb_marker_sum += cols.limb_shift_marker[limb].into();
            limb_shift += AB::Expr::from_usize(limb) * cols.limb_shift_marker[limb];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[limb]);
            for out_limb in 0..NUM_LIMBS {
                let src_limb = (out_limb + NUM_LIMBS - limb) % NUM_LIMBS;
                let carry_src = (out_limb + NUM_LIMBS - limb - 1) % NUM_LIMBS;
                when_limb_shift.assert_eq(
                    cols.a[out_limb],
                    cols.bit_shift_aux[src_limb] * bit_multiplier.clone()
                        + cols.bit_shift_carry[carry_src],
                );
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        for limb in 0..NUM_LIMBS {
            builder.assert_eq(
                cols.b[limb],
                cols.bit_shift_aux[limb] + cols.bit_shift_carry[limb] * carry_multiplier.clone(),
            );
            self.range_bus
                .send(cols.bit_shift_carry[limb], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(
                    cols.bit_shift_aux[limb],
                    AB::Expr::from_usize(BITMANIP_LIMB_BITS) - bit_shift.clone(),
                )
                .eval(builder, is_valid.clone());
        }

        // shamt = (width - idx) % width, encoded without a modulus by special
        // casing idx = 0 through the product of the two zero markers.
        let idx = limb_shift * AB::Expr::from_usize(BITMANIP_LIMB_BITS) + bit_shift;
        let is_idx_zero = cols.limb_shift_marker[0] * cols.bit_shift_marker[0];
        let immediate =
            (is_valid.clone() - is_idx_zero) * AB::Expr::from_usize(width_bits) - idx;

        let expected_opcode = AB::Expr::from_usize(BITMANIP_OFFSET + self.rori_opcode);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipRotateImmCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub imm: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipRotateImmExecutor<A, const NUM_LIMBS: usize> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipRotateImmFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA>
    for BitManipRotateImmExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 1]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BitManipRotateImmCoreRecord<NUM_LIMBS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BRotateImm({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_rotate_imm_opcode(local_opcode));
        let imm = instruction.c.as_canonical_u32();

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.imm = imm as u8;

        let (output, _) = run_rotate(false, &core_record.b, imm as u16);
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

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for BitManipRotateImmFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipRotateImmCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let imm = record.imm;

        let core_row: &mut BitManipRotateImmCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();
        let (a, idx) = run_rotate(false, &b, imm as u16);

        fill_rotate_aux(
            &mut core_row.bit_shift_marker,
            &mut core_row.limb_shift_marker,
            &mut core_row.bit_shift_carry,
            &mut core_row.bit_shift_aux,
            &b,
            idx,
            &self.range_checker_chip,
        );

        core_row.b = b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}
