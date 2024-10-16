use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::CastfOpcode;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    rv32im::adapters::RV32_REGISTER_NUM_LANES,
    system::program::Instruction,
};

// LIMB_SIZE is the size of the limbs in bits.
pub(crate) const LIMB_SIZE: usize = 8;
// the final limb has only 6 bits
pub(crate) const FINAL_LIMB_SIZE: usize = 6;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NewCastFCoreCols<T> {
    pub in_val: T,
    pub out_val: [T; RV32_REGISTER_NUM_LANES],
}

#[derive(Copy, Clone, Debug)]
pub struct NewCastFCoreAir {
    pub bus: VariableRangeCheckerBus, // to communicate with the range checker that checks that all limbs are < 2^LIMB_SIZE
    offset: usize,
}

impl<F: Field> BaseAir<F> for NewCastFCoreAir {
    fn width(&self) -> usize {
        NewCastFCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for NewCastFCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for NewCastFCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 1]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LANES]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &NewCastFCoreCols<_> = local_core.borrow();

        let intermed_val = cols
            .out_val
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &limb)| {
                acc + limb * AB::Expr::from_canonical_u32(1 << (i * LIMB_SIZE))
            });

        for i in 0..4 {
            self.bus
                .range_check(
                    cols.out_val[i],
                    match i {
                        0..=2 => LIMB_SIZE,
                        3 => FINAL_LIMB_SIZE,
                        _ => unreachable!(),
                    },
                )
                .eval(builder, AB::Expr::one());
        }

        AdapterAirContext {
            to_pc: None,
            reads: [[intermed_val]].into(),
            writes: [cols.out_val.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: AB::Expr::one(),
                opcode: AB::Expr::from_canonical_usize(CastfOpcode::CASTF as usize + self.offset),
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct NewCastFRecord<F> {
    pub in_val: F,
    pub out_val: [F; RV32_REGISTER_NUM_LANES],
}

#[derive(Debug)]
pub struct NewCastFCoreChip {
    pub air: NewCastFCoreAir,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl NewCastFCoreChip {
    pub fn new(range_checker_chip: Arc<VariableRangeCheckerChip>, offset: usize) -> Self {
        Self {
            air: NewCastFCoreAir {
                bus: range_checker_chip.bus(),
                offset,
            },
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for NewCastFCoreChip
where
    I::Reads: Into<[[F; 1]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LANES]; 1]>,
{
    type Record = NewCastFRecord<F>;
    type Air = NewCastFCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction.clone();

        assert_eq!(opcode - self.air.offset, CastfOpcode::CASTF as usize);

        let y = reads.into()[0][0];

        let x = NewCastF::solve(y.as_canonical_u32());
        for (i, limb) in x.iter().enumerate() {
            if i == 3 {
                self.range_checker_chip.add_count(*limb, FINAL_LIMB_SIZE);
            } else {
                self.range_checker_chip.add_count(*limb, LIMB_SIZE);
            }
        }

        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [x.map(F::from_canonical_u32)].into(),
        };

        let record = NewCastFRecord {
            in_val: y,
            out_val: x.map(F::from_canonical_u32),
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", CastfOpcode::CASTF)
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let cols: &mut NewCastFCoreCols<F> = row_slice.borrow_mut();
        cols.in_val = record.in_val;
        cols.out_val = record.out_val;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub struct NewCastF;
impl NewCastF {
    pub(super) fn solve(y: u32) -> [u32; RV32_REGISTER_NUM_LANES] {
        let mut x = [0; 4];
        for (i, limb) in x.iter_mut().enumerate() {
            *limb = (y >> (8 * i)) & 0xFF;
        }
        x
    }
}
