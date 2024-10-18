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
pub struct MillerDoubleCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl MillerDoubleCoreAir {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        max_limb_bits: usize,
        offset: usize,
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

        let mut x = Fp2::new(builder.clone());
        let mut y = Fp2::new(builder.clone());
        // λ = (3x^2) / (2y)
        let mut _3x2 = x.square().int_mul(3);
        let mut _2y = y.int_mul(2);
        let mut lambda = _3x2.div(&mut _2y);
        // x_2S = λ^2 - 2x
        let mut _2x = x.int_mul(2);
        let mut x_2s = lambda.square().sub(&mut _2x);
        x_2s.save();
        // y_2S = λ(x - x_2S) - y
        let mut y_2s = x.sub(&mut x_2s).mul(&mut lambda).sub(&mut y);
        y_2s.save();

        let mut b = lambda.int_mul(-1);
        b.save();
        let mut c = lambda.mul(&mut x).sub(&mut y);
        c.save();

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus: bus,
        };
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for MillerDoubleCoreAir {
    fn width(&self) -> usize {
        1 // TODO
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MillerDoubleCoreAir {}

impl<AB: AirBuilder, I> VmCoreAir<AB, I> for MillerDoubleCoreAir
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

pub struct MillerDoubleCoreChip {
    pub air: MillerDoubleCoreAir,
}

impl MillerDoubleCoreChip {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        max_limb_bits: usize,
        offset: usize,
    ) -> Self {
        let air = MillerDoubleCoreAir::new(
            modulus,
            num_limbs,
            limb_bits,
            range_checker,
            max_limb_bits,
            offset,
        );
        Self { air }
    }
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for MillerDoubleCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = MillerDoubleCoreAir;

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

#[cfg(test)]
mod tests {
    use afs_primitives::bigint::utils::{
        big_uint_mod_inverse, secp256k1_coord_prime, secp256k1_scalar_prime,
    };
    use ax_sdk::utils::create_seeded_rng;
    use axvm_instructions::EccOpcode;
    use num_bigint_dig::BigUint;
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, Field};
    use rand::Rng;

    use super::MillerDoubleCoreChip;
    use crate::{
        arch::{
            instructions::{ModularArithmeticOpcode, UsizeOpcode},
            testing::VmChipTestBuilder,
            VmChipWrapper,
        },
        rv32im::adapters::{Rv32VecHeapAdapterChip, RV32_REGISTER_NUM_LANES},
        system::program::Instruction,
        utils::biguint_to_limbs,
    };

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_miller_double() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let core = MillerDoubleCoreChip::new(
            secp256k1_scalar_prime(),
            NUM_LIMBS,
            LIMB_BITS,
            tester.memory_controller().borrow().range_checker.clone(),
            BabyBear::bits() - 2,
            EccOpcode::default_offset(),
        );
    }
}
