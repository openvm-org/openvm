use std::{array, borrow::BorrowMut, iter::repeat};

use afs_primitives::bigint::{
    check_carry_to_zero::get_carry_max_abs_and_bits,
    utils::{big_int_to_limbs, big_uint_sub},
    CanonicalUint, DefaultLimbConfig, OverflowInt,
};
use afs_stark_backend::rap::{get_air_name, AnyRap};
use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{
    columns::{ModularMultDivAuxCols, ModularMultDivCols, ModularMultDivIoCols},
    ModularMultDivChip,
};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        MachineChip,
    },
    memory::MemoryHeapDataIoCols,
    utils::limbs_to_biguint,
};

impl<F: PrimeField32, const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    MachineChip<F> for ModularMultDivChip<F, CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let height = self.data.len();
        let height = height.next_power_of_two();

        let blank_row = vec![F::zero(); ModularMultDivCols::<F, CARRY_LIMBS, NUM_LIMBS>::width()];
        let mut rows = vec![blank_row; height];

        for (i, record) in self.data.iter().enumerate() {
            let row = &mut rows[i];
            let cols: &mut ModularMultDivCols<F, CARRY_LIMBS, NUM_LIMBS> = row[..].borrow_mut();
            cols.io = ModularMultDivIoCols {
                from_state: record.from_state.map(F::from_canonical_usize),
                x: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(record.x_array_read.clone()),
                y: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(record.y_array_read.clone()),
                z: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(record.z_array_write.clone()),
            };
            let x = limbs_to_biguint(
                &record
                    .x_array_read
                    .data_read
                    .data
                    .map(|x| x.as_canonical_u32()),
                LIMB_SIZE,
            );
            let y = limbs_to_biguint(
                &record
                    .y_array_read
                    .data_read
                    .data
                    .map(|x| x.as_canonical_u32()),
                LIMB_SIZE,
            );
            let r = limbs_to_biguint(
                &record
                    .z_array_write
                    .data_write
                    .data
                    .map(|x| x.as_canonical_u32()),
                LIMB_SIZE,
            );
            let is_mult = match ModularArithmeticOpcode::from_usize(record.instruction.opcode) {
                ModularArithmeticOpcode::MUL => true,
                ModularArithmeticOpcode::DIV => false,
                _ => unreachable!(),
            };

            if is_mult {
                self.generate_aux_cols_mult(cols.aux.borrow_mut(), x, y, r);
            } else {
                self.generate_aux_cols_div(cols.aux.borrow_mut(), x, y, r);
            }

            cols.aux.is_valid = F::one();
            cols.aux.read_x_aux_cols =
                aux_cols_factory.make_heap_read_aux_cols(record.x_array_read.clone());
            cols.aux.read_y_aux_cols =
                aux_cols_factory.make_heap_read_aux_cols(record.y_array_read.clone());
            cols.aux.write_z_aux_cols =
                aux_cols_factory.make_heap_write_aux_cols(record.z_array_write.clone());
            cols.aux.is_mult = F::from_bool(is_mult);
        }

        RowMajorMatrix::new(
            rows.concat(),
            ModularMultDivCols::<F, CARRY_LIMBS, NUM_LIMBS>::width(),
        )
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        ModularMultDivCols::<F, CARRY_LIMBS, NUM_LIMBS>::width()
    }
}

impl<F: PrimeField32, const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    ModularMultDivChip<F, CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
    fn generate_aux_cols_mult(
        &self,
        aux: &mut ModularMultDivAuxCols<F, CARRY_LIMBS, NUM_LIMBS>,
        x: BigUint,
        y: BigUint,
        r: BigUint,
    ) {
        let q = big_uint_sub(x.clone() * y.clone(), r.clone());
        let q = q / BigInt::from_biguint(Sign::Plus, self.modulus.clone());
        self.generate_aux_cols(aux, x, y, r, q, true);
    }
    fn generate_aux_cols_div(
        &self,
        aux: &mut ModularMultDivAuxCols<F, CARRY_LIMBS, NUM_LIMBS>,
        x: BigUint,
        y: BigUint,
        r: BigUint,
    ) {
        let q = big_uint_sub(y.clone() * r.clone(), x.clone());
        let q = q / BigInt::from_biguint(Sign::Plus, self.modulus.clone());
        self.generate_aux_cols(aux, x, y, r, q, false);
    }
    fn generate_aux_cols(
        &self,
        aux: &mut ModularMultDivAuxCols<F, CARRY_LIMBS, NUM_LIMBS>,
        x: BigUint,
        y: BigUint,
        r: BigUint,
        q: BigInt,
        is_mult: bool,
    ) {
        // Quotient and result can be smaller, but padding to the desired length.
        let q_limbs: Vec<isize> = big_int_to_limbs(&q, LIMB_SIZE)
            .iter()
            .chain(repeat(&0))
            .take(NUM_LIMBS)
            .copied()
            .collect();
        for &q in q_limbs.iter() {
            self.range_checker_chip
                .add_count((q + (1 << LIMB_SIZE)) as u32, LIMB_SIZE + 1);
        }
        aux.q = array::from_fn(|i| {
            if q_limbs[i] >= 0 {
                F::from_canonical_usize(q_limbs[i].unsigned_abs())
            } else {
                F::from_canonical_usize(q_limbs[i].unsigned_abs()) * F::neg_one()
            }
        });

        let x: OverflowInt<isize> =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&x, Some(NUM_LIMBS)).into();
        let y: OverflowInt<isize> =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&y, Some(NUM_LIMBS)).into();
        let r: OverflowInt<isize> =
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(&r, Some(NUM_LIMBS)).into();
        let p: OverflowInt<isize> = CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(
            &self.modulus,
            Some(NUM_LIMBS),
        )
        .into();

        let q_overflow = OverflowInt {
            limbs: q_limbs,
            max_overflow_bits: LIMB_SIZE + 1,
            limb_max_abs: (1 << LIMB_SIZE),
        };

        let expr: OverflowInt<isize> = if is_mult { x * y - r } else { r * y - x } - p * q_overflow;
        let carries = expr.calculate_carries(LIMB_SIZE);
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, LIMB_SIZE);

        for (i, &carry) in carries.iter().enumerate() {
            self.range_checker_chip
                .add_count((carry + carry_min_abs as isize) as u32, carry_bits);
            let carry_f = F::from_canonical_usize(carry.unsigned_abs());
            aux.carries[i] = if carry >= 0 {
                carry_f
            } else {
                carry_f * F::neg_one()
            };
        }
    }
}
