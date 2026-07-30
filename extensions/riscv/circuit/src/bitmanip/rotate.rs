use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

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
    core::{is_rotate_reg_opcode, BITMANIP_OFFSET, ROL, ROLW},
    BITMANIP_LIMB_BITS,
};

/// Register-register rotates over `NUM_LIMBS` u16 limbs (ROL/ROR at 4 limbs,
/// ROLW/RORW at 2 limbs behind the sign-extending W adapter).
///
/// The recombination machinery always computes a LEFT rotate by the
/// marker-encoded amount `idx`; a right rotate by `s` is a left rotate by
/// `(width - s) % width`, so the direction only changes how `idx` is bound to
/// the rs2 amount:
///   ROL: rs2_low ≡ idx  (mod width),  i.e. (c[0] - idx) / width is a small int
///   ROR: rs2_low ≡ -idx (mod width),  i.e. (c[0] + idx) / width is a small int
///
/// Each `b` limb splits at bit `beta = idx % 16` into `aux` (low `16 - beta`
/// bits) and `carry` (high `beta` bits); output limb `j` recombines
/// `aux[(j - L) % N] * 2^beta + carry[(j - L - 1) % N]` where `L = idx / 16`.
/// Unlike a shift, every output limb has a carry-in (the rotate wraps), and
/// the additive recombination bounds each output limb below `2^16`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipRotateCoreCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub opcode_rol_flag: T,
    pub opcode_ror_flag: T,
    pub bit_shift_marker: [T; BITMANIP_LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipRotateCoreCols<u8, NUM_LIMBS>)]
pub struct BitManipRotateCoreAir<const NUM_LIMBS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    /// Local opcodes of the left/right rotate this instantiation serves.
    pub rol_opcode: usize,
    pub ror_opcode: usize,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for BitManipRotateCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        BitManipRotateCoreCols::<F, NUM_LIMBS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F>
    for BitManipRotateCoreAir<NUM_LIMBS>
{
}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for BitManipRotateCoreAir<NUM_LIMBS>
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
        let cols: &BitManipRotateCoreCols<_, NUM_LIMBS> = local_core.borrow();
        let width_bits = NUM_LIMBS * BITMANIP_LIMB_BITS;

        builder.assert_bool(cols.opcode_rol_flag);
        builder.assert_bool(cols.opcode_ror_flag);
        let is_valid: AB::Expr = cols.opcode_rol_flag + cols.opcode_ror_flag;
        builder.assert_bool(is_valid.clone());

        // One-hot markers encode the normalized left-rotate amount
        // idx = 16 * limb_shift + bit_shift.
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
        builder.assert_eq(bit_marker_sum, is_valid.clone());

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

        // Split each b limb at the bit shift; the recombination above is then
        // exact. Vacuous on padding rows (all columns zero).
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

        // Bind idx to the rotate amount rs2 % width per direction. The
        // quotient is range checked so the congruence holds exactly; idx is in
        // [0, width) by construction of the markers.
        let idx = limb_shift * AB::Expr::from_usize(BITMANIP_LIMB_BITS) + bit_shift;
        let width_inv = AB::F::from_usize(width_bits).inverse();
        let quotient_bits = BITMANIP_LIMB_BITS - width_bits.ilog2() as usize;
        self.range_bus
            .range_check((cols.c[0] - idx.clone()) * width_inv, quotient_bits)
            .eval(builder, cols.opcode_rol_flag);
        self.range_bus
            .range_check((cols.c[0] + idx) * width_inv, quotient_bits + 1)
            .eval(builder, cols.opcode_ror_flag);

        let expected_opcode = cols.opcode_rol_flag * AB::Expr::from_usize(self.rol_opcode)
            + cols.opcode_ror_flag * AB::Expr::from_usize(self.ror_opcode)
            + is_valid.clone() * AB::Expr::from_usize(BITMANIP_OFFSET);

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
        BITMANIP_OFFSET
    }
}

/// Returns the rotated limbs and the normalized left-rotate amount `idx`.
pub(crate) fn run_rotate<const NUM_LIMBS: usize>(
    is_rol: bool,
    b: &[u16; NUM_LIMBS],
    c0: u16,
) -> ([u16; NUM_LIMBS], usize) {
    let width = NUM_LIMBS * BITMANIP_LIMB_BITS;
    debug_assert!(width <= 64);
    let s = (c0 as usize) % width;
    let idx = if is_rol { s } else { (width - s) % width };

    let value = b
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, limb)| acc | ((*limb as u64) << (16 * i)));
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let rotated = if idx == 0 {
        value
    } else {
        ((value << idx) | (value >> (width - idx))) & mask
    };
    (
        array::from_fn(|i| ((rotated >> (16 * i)) & 0xffff) as u16),
        idx,
    )
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipRotateCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub c: [u16; NUM_LIMBS],
    pub is_rol: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipRotateExecutor<A, const NUM_LIMBS: usize> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipRotateFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA>
    for BitManipRotateExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 2]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BitManipRotateCoreRecord<NUM_LIMBS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BRotate({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_rotate_reg_opcode(local_opcode));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.is_rol = matches!(local_opcode, ROL | ROLW) as u8;

        let (output, _) = run_rotate(core_record.is_rol != 0, &core_record.b, core_record.c[0]);
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

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for BitManipRotateFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipRotateCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let c = record.c;
        let is_rol = record.is_rol != 0;

        let core_row: &mut BitManipRotateCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();
        let (a, idx) = run_rotate(is_rol, &b, c[0]);

        fill_rotate_aux(
            &mut core_row.bit_shift_marker,
            &mut core_row.limb_shift_marker,
            &mut core_row.bit_shift_carry,
            &mut core_row.bit_shift_aux,
            &b,
            idx,
            &self.range_checker_chip,
        );

        let width_bits = NUM_LIMBS * BITMANIP_LIMB_BITS;
        let quotient_bits = BITMANIP_LIMB_BITS - width_bits.ilog2() as usize;
        if is_rol {
            self.range_checker_chip
                .add_count((c[0] as usize - idx) as u32 / width_bits as u32, quotient_bits);
        } else {
            self.range_checker_chip.add_count(
                (c[0] as usize + idx) as u32 / width_bits as u32,
                quotient_bits + 1,
            );
        }

        core_row.opcode_ror_flag = F::from_bool(!is_rol);
        core_row.opcode_rol_flag = F::from_bool(is_rol);
        core_row.c = c.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}

/// Fills the shared rotate witness columns (markers plus the carry/aux split
/// of each `b` limb at `idx % 16`) and issues their range checks.
pub(super) fn fill_rotate_aux<F: PrimeField32, const NUM_LIMBS: usize>(
    bit_shift_marker: &mut [F; BITMANIP_LIMB_BITS],
    limb_shift_marker: &mut [F; NUM_LIMBS],
    bit_shift_carry: &mut [F; NUM_LIMBS],
    bit_shift_aux: &mut [F; NUM_LIMBS],
    b: &[u16; NUM_LIMBS],
    idx: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let bit_shift = idx % BITMANIP_LIMB_BITS;
    let limb_shift = idx / BITMANIP_LIMB_BITS;
    *bit_shift_marker = [F::ZERO; BITMANIP_LIMB_BITS];
    bit_shift_marker[bit_shift] = F::ONE;
    *limb_shift_marker = [F::ZERO; NUM_LIMBS];
    limb_shift_marker[limb_shift] = F::ONE;

    let aux_bits = BITMANIP_LIMB_BITS - bit_shift;
    for (limb, &b_limb) in b.iter().enumerate() {
        let limb_u32 = b_limb as u32;
        let aux = limb_u32 & ((1u32 << aux_bits) - 1);
        let carry = limb_u32 >> aux_bits;
        bit_shift_aux[limb] = F::from_u32(aux);
        bit_shift_carry[limb] = F::from_u32(carry);
        range_checker.add_count(carry, bit_shift);
        range_checker.add_count(aux, aux_bits);
    }
}
