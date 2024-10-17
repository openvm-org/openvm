use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::xor::{bus::XorBus, lookup::XorLookupChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use strum::IntoEnumIterator;

use crate::{
    arch::{
        instructions::{LessThanOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct LessThanCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub cmp_result: T,

    pub opcode_slt_flag: T,
    pub opcode_sltu_flag: T,

    // Most significant limb of b and c respectively as a field element, will be range
    // checked to be within [-128, 127). Field xor_res is the result of (b_msb_f + 128)
    // ^ (c_msb_f + 128) if signed and b_msb_f ^ c_msb_f else, used to range check
    // b_msb_f and c_msb_f.
    pub b_msb_f: T,
    pub c_msb_f: T,
    pub xor_res: T,

    // 1 at the most significant index i such that b[i] != c[i], otherwise 0. If such
    // an i exists, diff_val = c[i] - b[i]
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug)]
pub struct LessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: XorBus,
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

        let is_valid = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let b = &cols.b;
        let c = &cols.c;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::zero();

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
            }) * (AB::Expr::from_canonical_u8(2) * cols.cmp_result - AB::Expr::one());
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero((AB::Expr::one() - prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(AB::Expr::one() - prefix_sum)
            .assert_zero(cols.cmp_result);

        self.bus
            .send(
                cols.b_msb_f
                    + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag,
                cols.c_msb_f
                    + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag,
                cols.xor_res,
            )
            .eval(builder, is_valid.clone());
        self.bus
            .send(
                cols.diff_val - AB::Expr::one(),
                cols.diff_val - AB::Expr::one(),
                AB::F::zero(),
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = flags
            .iter()
            .zip(LessThanOpcode::iter())
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
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
}

#[derive(Clone, Debug)]
pub struct LessThanCoreRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub opcode: LessThanOpcode,
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub cmp_result: T,
    pub b_msb_f: T,
    pub c_msb_f: T,
    pub xor_res: T,
    pub diff_val: T,
    pub diff_idx: usize,
}

#[derive(Debug)]
pub struct LessThanCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: LessThanCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> LessThanCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>, offset: usize) -> Self {
        Self {
            air: LessThanCoreAir {
                bus: xor_lookup_chip.bus(),
                offset,
            },
            xor_lookup_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, I> for LessThanCoreChip<NUM_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = LessThanCoreRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Air = LessThanCoreAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let less_than_opcode = LessThanOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (cmp_result, diff_idx, b_sign, c_sign) =
            solve_less_than::<NUM_LIMBS, LIMB_BITS>(less_than_opcode, &b, &c);

        // xor_res is the result of (b_msb_f + 128) ^ (c_msb_f + 128) if signed,
        // b_msb_f ^ c_msb_f if not
        let (b_msb_f, b_msb_xor) = if b_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1] - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1]
                    + (((less_than_opcode == LessThanOpcode::SLT) as u32) << (LIMB_BITS - 1)),
            )
        };
        let (c_msb_f, c_msb_xor) = if c_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - c[NUM_LIMBS - 1]),
                c[NUM_LIMBS - 1] - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(c[NUM_LIMBS - 1]),
                c[NUM_LIMBS - 1]
                    + (((less_than_opcode == LessThanOpcode::SLT) as u32) << (LIMB_BITS - 1)),
            )
        };
        let xor_res = self.xor_lookup_chip.request(b_msb_xor, c_msb_xor);

        let diff_val = if diff_idx == (NUM_LIMBS - 1) {
            if cmp_result {
                c_msb_f - b_msb_f
            } else {
                b_msb_f - c_msb_f
            }
            .as_canonical_u32()
        } else if cmp_result {
            c[diff_idx] - b[diff_idx]
        } else {
            b[diff_idx] - c[diff_idx]
        };

        // TODO: update XorLookupChip to either be BitwiseOperation or range check
        self.xor_lookup_chip.request(diff_val - 1, diff_val - 1);

        let mut writes = [0u32; NUM_LIMBS];
        writes[0] = cmp_result as u32;

        let output = AdapterRuntimeContext::without_pc([writes.map(F::from_canonical_u32)]);
        let record = LessThanCoreRecord {
            opcode: less_than_opcode,
            b: b.map(F::from_canonical_u32),
            c: c.map(F::from_canonical_u32),
            cmp_result: F::from_bool(cmp_result),
            b_msb_f,
            c_msb_f,
            xor_res: F::from_canonical_u32(xor_res),
            diff_val: F::from_canonical_u32(diff_val),
            diff_idx,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LessThanOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut LessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = row_slice.borrow_mut();
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.cmp_result = record.cmp_result;
        row_slice.b_msb_f = record.b_msb_f;
        row_slice.c_msb_f = record.c_msb_f;
        row_slice.xor_res = record.xor_res;
        row_slice.diff_val = record.diff_val;
        row_slice.opcode_slt_flag = F::from_bool(record.opcode == LessThanOpcode::SLT);
        row_slice.opcode_sltu_flag = F::from_bool(record.opcode == LessThanOpcode::SLTU);
        row_slice.diff_marker = array::from_fn(|i| F::from_bool(i == record.diff_idx));
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
pub(super) fn solve_less_than<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: LessThanOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
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
