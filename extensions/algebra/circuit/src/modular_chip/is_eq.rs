use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
};

use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    is_equal_array::{IsEqArrayIo, IsEqArraySubAir},
    SubAir, TraceSubRowGenerator,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
// Given two numbers b and c, we want to prove that a) b == c or b != c, depending on
// result of cmp_result and b) b, c < N for some modulus N that is passed into the AIR
// at runtime (i.e. when chip is instantiated).

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ModularIsEqualCoreCols<T, const READ_LIMBS: usize> {
    pub is_valid: T,
    pub is_setup: T,
    pub b: [T; READ_LIMBS],
    pub c: [T; READ_LIMBS],
    pub cmp_result: T,

    // Auxiliary columns for subair EQ comparison between b and c.
    pub eq_marker: [T; READ_LIMBS],

    // Auxiliary columns to ensure both b and c are smaller than modulus N. Let b_diff_idx be
    // an index such that b[b_diff_idx] < N[b_diff_idx] and b[i] = N[i] for all i > b_diff_idx,
    // where larger indices correspond to more significant limbs. Such an index exists iff b < N.
    // Define c_diff_idx analogously. Then let b_lt_diff = N[b_diff_idx] - b[b_diff_idx] and
    // c_lt_diff = N[c_diff_idx] - c[c_diff_idx], where both must be in [0, 2^LIMB_BITS).
    //
    // To constrain the above, we will use lt_marker, which will indicate where b_diff_idx and
    // c_diff_idx are. Set lt_marker[b_diff_idx] = 1, lt_marker[c_diff_idx] = c_lt_mark, and 0
    // everywhere else. If b_diff_idx == c_diff_idx then c_lt_mark = 1, else c_lt_mark = 2. The
    // purpose of c_lt_mark is to handle the edge case where b_diff_idx == c_diff_idx (because
    // we cannot set lt_marker[b_diff_idx] to 1 and 2 at the same time).
    pub lt_marker: [T; READ_LIMBS],
    pub b_lt_diff: T,
    pub c_lt_diff: T,
    pub c_lt_mark: T,
}

#[derive(Clone, Debug)]
pub struct ModularIsEqualCoreAir<
    const READ_LIMBS: usize,
    const WRITE_LIMBS: usize,
    const LIMB_BITS: usize,
> {
    pub bus: BitwiseOperationLookupBus,
    pub subair: IsEqArraySubAir<READ_LIMBS>,
    pub modulus_limbs: [u32; READ_LIMBS],
    pub offset: usize,
}

impl<const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize>
    ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
{
    pub fn new(modulus: BigUint, bus: BitwiseOperationLookupBus, offset: usize) -> Self {
        let mod_vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        assert!(mod_vec.len() <= READ_LIMBS);
        let modulus_limbs = array::from_fn(|i| {
            if i < mod_vec.len() {
                mod_vec[i] as u32
            } else {
                0
            }
        });
        Self {
            bus,
            subair: IsEqArraySubAir::<READ_LIMBS>,
            modulus_limbs,
            offset,
        }
    }
}

impl<F: Field, const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ModularIsEqualCoreCols::<F, READ_LIMBS>::width()
    }
}
impl<F: Field, const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize>
    BaseAirWithPublicValues<F> for ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreAir<AB, I> for ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; READ_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; WRITE_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &ModularIsEqualCoreCols<_, READ_LIMBS> = local_core.borrow();

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.is_setup);
        builder.when(cols.is_setup).assert_one(cols.is_valid);
        builder.assert_bool(cols.cmp_result);

        // Constrain that either b == c or b != c, depending on the value of cmp_result.
        let eq_subair_io = IsEqArrayIo {
            x: cols.b.map(Into::into),
            y: cols.c.map(Into::into),
            out: cols.cmp_result.into(),
            condition: cols.is_valid - cols.is_setup,
        };
        self.subair.eval(builder, (eq_subair_io, cols.eq_marker));

        // Constrain that auxiliary columns lt_columns and c_lt_mark are as defined above.
        // When c_lt_mark is 1, lt_marker should have exactly one index i where lt_marker[i]
        // is 1, and be 0 elsewhere. When c_lt_mark is 2, lt_marker[i] should have an
        // additional index j such that lt_marker[j] is 2. To constrain this:
        //
        // * When c_lt_mark = 1 the sum of all lt_marker[i] must be 1
        // * When c_lt_mark = 2 the sum of lt_marker[i] * (lt_marker[i] - 1) must be 2.
        //   Additionally, the sum of all lt_marker[i] must be 3.
        //
        // All this doesn't apply when is_setup.
        let lt_marker_sum = cols
            .lt_marker
            .iter()
            .fold(AB::Expr::ZERO, |acc, x| acc + *x);
        let lt_marker_one_check_sum = cols
            .lt_marker
            .iter()
            .fold(AB::Expr::ZERO, |acc, x| acc + (*x) * (*x - AB::F::ONE));

        // Constrain that c_lt_mark is either 1 or 2.
        builder
            .when(cols.is_valid - cols.is_setup)
            .assert_bool(cols.c_lt_mark - AB::F::ONE);

        // If c_lt_mark is 1, then lt_marker_sum is 1
        builder
            .when(cols.is_valid - cols.is_setup)
            .when_ne(cols.c_lt_mark, AB::F::from_canonical_u8(2))
            .assert_one(lt_marker_sum.clone());

        // If c_lt_mark is 2, then lt_marker_sum is 3
        builder
            .when(cols.is_valid - cols.is_setup)
            .when_ne(cols.c_lt_mark, AB::F::ONE)
            .assert_eq(lt_marker_sum.clone(), AB::F::from_canonical_u8(3));

        // This constraint, along with the constraint (below) that lt_marker[i] is 0, 1, or 2,
        // ensures that lt_marker has exactly one 2.
        builder.when_ne(cols.c_lt_mark, AB::F::ONE).assert_eq(
            lt_marker_one_check_sum,
            cols.is_valid * AB::F::from_canonical_u8(2),
        );

        // Handle the setup row constraints.
        // When is_setup = 1, constrain c_lt_mark = 2 and lt_marker_sum = 2
        // This ensures that lt_marker has exactly one 2 and the remaining entries are 0.
        // Since lt_marker has no 1, we will end up constraining that b[i] = N[i] for all i
        // instead of just for i > b_diff_idx.
        builder
            .when(cols.is_setup)
            .assert_eq(cols.c_lt_mark, AB::F::from_canonical_u8(2));
        builder
            .when(cols.is_setup)
            .assert_eq(lt_marker_sum.clone(), AB::F::from_canonical_u8(2));

        // Constrain that b, c < N (i.e. modulus).
        let modulus = self.modulus_limbs.map(AB::F::from_canonical_u32);
        let mut prefix_sum = AB::Expr::ZERO;

        for i in (0..READ_LIMBS).rev() {
            prefix_sum += cols.lt_marker[i].into();
            builder.assert_zero(
                cols.lt_marker[i]
                    * (cols.lt_marker[i] - AB::F::ONE)
                    * (cols.lt_marker[i] - cols.c_lt_mark),
            );

            // Constrain b < N.
            // First, we constrain b[i] = N[i] for i > b_diff_idx.
            // We do this by constraining that b[i] = N[i] when prefix_sum is not 1 or
            // lt_marker_sum.
            //  - If is_setup = 0, then lt_marker_sum is either 1 or 3. In this case, prefix_sum is
            //    0, 1, 2, or 3. It can be verified by casework that i > b_diff_idx iff prefix_sum
            //    is not 1 or lt_marker_sum.
            //  - If is_setup = 1, then we want to constrain b[i] = N[i] for all i. In this case,
            //    lt_marker_sum is 2 and prefix_sum is 0 or 2. So we constrain b[i] = N[i] when
            //    prefix_sum is not 1, which works.
            builder
                .when_ne(prefix_sum.clone(), AB::F::ONE)
                .when_ne(prefix_sum.clone(), lt_marker_sum.clone() - cols.is_setup)
                .assert_eq(cols.b[i], modulus[i]);
            // Note that lt_marker[i] is either 0, 1, or 2 and lt_marker[i] being 1 indicates b[i] <
            // N[i] (i.e. i == b_diff_idx).
            builder
                .when_ne(cols.lt_marker[i], AB::F::ZERO)
                .when_ne(cols.lt_marker[i], AB::F::from_canonical_u8(2))
                .assert_eq(AB::Expr::from(modulus[i]) - cols.b[i], cols.b_lt_diff);

            // Constrain c < N.
            // First, we constrain c[i] = N[i] for i > c_diff_idx.
            // We do this by constraining that c[i] = N[i] when prefix_sum is not c_lt_mark or
            // lt_marker_sum. It can be verified by casework that i > c_diff_idx iff
            // prefix_sum is not c_lt_mark or lt_marker_sum.
            builder
                .when_ne(prefix_sum.clone(), cols.c_lt_mark)
                .when_ne(prefix_sum.clone(), lt_marker_sum.clone())
                .assert_eq(cols.c[i], modulus[i]);
            // Note that lt_marker[i] is either 0, 1, or 2 and lt_marker[i] being c_lt_mark
            // indicates c[i] < N[i] (i.e. i == c_diff_idx). Since c_lt_mark is 1 or 2,
            // we have {0, 1, 2} \ {0, 3 - c_lt_mark} = {c_lt_mark}.
            builder
                .when_ne(cols.lt_marker[i], AB::F::ZERO)
                .when_ne(
                    cols.lt_marker[i],
                    AB::Expr::from_canonical_u8(3) - cols.c_lt_mark,
                )
                .assert_eq(AB::Expr::from(modulus[i]) - cols.c[i], cols.c_lt_diff);
        }

        // Check that b_lt_diff and c_lt_diff are positive
        self.bus
            .send_range(
                cols.b_lt_diff - AB::Expr::ONE,
                cols.c_lt_diff - AB::Expr::ONE,
            )
            .eval(builder, cols.is_valid - cols.is_setup);

        let expected_opcode = AB::Expr::from_canonical_usize(self.offset)
            + cols.is_setup
                * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize)
            + (AB::Expr::ONE - cols.is_setup)
                * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::IS_EQ as usize);
        let mut a: [AB::Expr; WRITE_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModularIsEqualCoreRecord<T, const READ_LIMBS: usize> {
    #[serde(with = "BigArray")]
    pub b: [T; READ_LIMBS],
    #[serde(with = "BigArray")]
    pub c: [T; READ_LIMBS],
    pub cmp_result: T,
    #[serde(with = "BigArray")]
    pub eq_marker: [T; READ_LIMBS],
    pub b_diff_idx: usize,
    pub c_diff_idx: usize,
    pub is_setup: bool,
}

pub struct ModularIsEqualCoreChip<
    const READ_LIMBS: usize,
    const WRITE_LIMBS: usize,
    const LIMB_BITS: usize,
> {
    pub air: ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize>
    ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
{
    pub fn new(
        modulus: BigUint,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        Self {
            air: ModularIsEqualCoreAir::new(modulus, bitwise_lookup_chip.bus(), offset),
            bitwise_lookup_chip,
        }
    }
}

impl<
        F: PrimeField32,
        I: VmAdapterInterface<F>,
        const READ_LIMBS: usize,
        const WRITE_LIMBS: usize,
        const LIMB_BITS: usize,
    > VmCoreChip<F, I> for ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; READ_LIMBS]; 2]>,
    I::Writes: From<[[F; WRITE_LIMBS]; 1]>,
{
    type Record = ModularIsEqualCoreRecord<F, READ_LIMBS>;
    type Air = ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let data: [[F; READ_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (b_cmp, b_diff_idx) = run_unsigned_less_than::<READ_LIMBS>(&b, &self.air.modulus_limbs);
        let (c_cmp, c_diff_idx) = run_unsigned_less_than::<READ_LIMBS>(&c, &self.air.modulus_limbs);
        let is_setup = instruction.opcode.local_opcode_idx(self.air.offset)
            == Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize;

        if !is_setup {
            assert!(b_cmp, "{:?} >= {:?}", b, self.air.modulus_limbs);
        }
        assert!(c_cmp, "{:?} >= {:?}", c, self.air.modulus_limbs);
        if !is_setup {
            self.bitwise_lookup_chip.request_range(
                self.air.modulus_limbs[b_diff_idx] - b[b_diff_idx] - 1,
                self.air.modulus_limbs[c_diff_idx] - c[c_diff_idx] - 1,
            );
        }

        let mut eq_marker = [F::ZERO; READ_LIMBS];
        let mut cmp_result = F::ZERO;
        self.air
            .subair
            .generate_subrow((&data[0], &data[1]), (&mut eq_marker, &mut cmp_result));

        let mut writes = [F::ZERO; WRITE_LIMBS];
        writes[0] = cmp_result;

        let output = AdapterRuntimeContext::without_pc([writes]);
        let record = ModularIsEqualCoreRecord {
            is_setup,
            b: data[0],
            c: data[1],
            cmp_result,
            eq_marker,
            b_diff_idx,
            c_diff_idx,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32ModularArithmeticOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut ModularIsEqualCoreCols<_, READ_LIMBS> = row_slice.borrow_mut();
        row_slice.is_valid = F::ONE;
        row_slice.is_setup = F::from_bool(record.is_setup);
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.cmp_result = record.cmp_result;

        row_slice.eq_marker = record.eq_marker;

        if !record.is_setup {
            row_slice.b_lt_diff = F::from_canonical_u32(self.air.modulus_limbs[record.b_diff_idx])
                - record.b[record.b_diff_idx];
        }
        row_slice.c_lt_diff = F::from_canonical_u32(self.air.modulus_limbs[record.c_diff_idx])
            - record.c[record.c_diff_idx];
        row_slice.c_lt_mark = if record.b_diff_idx == record.c_diff_idx {
            F::ONE
        } else {
            F::from_canonical_u8(2)
        };
        row_slice.lt_marker = from_fn(|i| {
            if i == record.b_diff_idx {
                F::ONE
            } else if i == record.c_diff_idx {
                row_slice.c_lt_mark
            } else {
                F::ZERO
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// Returns (cmp_result, diff_idx)
pub(super) fn run_unsigned_less_than<const NUM_LIMBS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> (bool, usize) {
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return (x[i] < y[i], i);
        }
    }
    (false, NUM_LIMBS)
}
