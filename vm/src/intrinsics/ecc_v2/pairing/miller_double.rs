use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{Field, PrimeField32};

use super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct MillerDoubleAir {
    pub expr: FieldExpr,
}

impl MillerDoubleAir {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
        max_limb_bits: usize,
    ) -> Self {
        assert!(modulus.bits() <= num_limbs * limb_bits);
        let bus = range_checker.bus();
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_bits,
            bus.index,
            bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(
            modulus,
            limb_bits,
            num_limbs,
            range_checker.range_max_bits(),
            max_limb_bits,
        );
        let builder = Rc::new(RefCell::new(builder));

        let x = Fp2::new(builder.clone());
        let y = Fp2::new(builder.clone());
        // TODO:
        // λ = (3x^2) / (2y)
        // x_2S = λ^2 - 2x
        // y_2S = λ(x - x_2S) - y

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus: bus.index,
            range_max_bits: bus.range_max_bits,
        };
        Self { expr }
    }
}

impl<F: Field> BaseAir<F> for MillerDoubleAir {
    fn width(&self) -> usize {
        1 // TODO
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MillerDoubleAir {}

impl<AB: AirBuilder, I> VmCoreAir<AB, I> for MillerDoubleAir
where
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<Vec<AB::Expr>>,
    I::Writes: From<Vec<AB::Expr>>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
    }
}

pub struct MillerDoubleChip {
    pub air: MillerDoubleAir,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for MillerDoubleChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = MillerDoubleAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        // Input: EcPoint<Fp2>, so total 4 field elements.

        todo!()
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "MillerDoubleAndAdd".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
