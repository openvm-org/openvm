use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_primitives::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use ax_ecc_primitives::field_expression::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExprCols, FieldVariable, SymbolicExpr,
};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::instruction::Instruction;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{Rv32ModularArithmeticOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray,
        MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};

/// The number of limbs and limb bits are determined at runtime.
#[derive(Clone)]
pub struct ModularMulDivCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl ModularMulDivCoreAir {
    pub fn new(
        config: ExprBuilderConfig,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
    ) -> Self {
        config.check_valid();

        let builder = ExprBuilder::new(config, range_bus.range_max_bits);
        let builder = Rc::new(RefCell::new(builder));
        let x = ExprBuilder::new_input(builder.clone());
        let y = ExprBuilder::new_input(builder.clone());
        let (z_idx, z) = builder.borrow_mut().new_var();
        let z = FieldVariable::from_var(builder.clone(), z);
        let is_mul_flag = builder.borrow_mut().new_flag();
        let is_div_flag = builder.borrow_mut().new_flag();
        // constraint is x * y = z, or z * y = x
        let lvar = FieldVariable::select(is_mul_flag, &x, &z);
        let rvar = FieldVariable::select(is_mul_flag, &z, &x);
        let constraint = lvar * y.clone() - rvar;
        builder.borrow_mut().set_constraint(z_idx, constraint.expr);
        let compute = SymbolicExpr::Select(
            is_mul_flag,
            Box::new(x.expr.clone() * y.expr.clone()),
            Box::new(SymbolicExpr::Select(
                is_div_flag,
                Box::new(x.expr.clone() / y.expr.clone()),
                Box::new(x.expr.clone()),
            )),
        );
        builder.borrow_mut().set_compute(z_idx, compute);

        let builder = builder.borrow().clone();

        let expr = FieldExpr::new(builder, range_bus, true);
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for ModularMulDivCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for ModularMulDivCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for ModularMulDivCoreAir
where
    I: VmAdapterInterface<AB::Expr>,
    AdapterAirContext<AB::Expr, I>:
        From<AdapterAirContext<AB::Expr, DynAdapterInterface<AB::Expr>>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        assert_eq!(local.len(), BaseAir::<AB::F>::width(&self.expr));
        self.expr.eval(builder, local);

        let FieldExprCols {
            is_valid,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), 2);
        assert_eq!(vars.len(), 1);
        assert_eq!(flags.len(), 2);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[0].iter().map(|x| (*x).into()).collect();

        // Attention: we multiply in the setup case, hence flags[0] (is_mul_flag) does NOT imply that is_setup is false!
        let local_opcode_idx = flags[0]
            * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::MUL as usize)
            + flags[1] * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::DIV as usize)
            + (AB::Expr::ONE - flags[0] - flags[1])
                * AB::Expr::from_canonical_usize(
                    Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
                );

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: local_opcode_idx + AB::Expr::from_canonical_usize(self.offset),
        };

        let ctx: AdapterAirContext<_, DynAdapterInterface<_>> = AdapterAirContext {
            to_pc: None,
            reads: reads.into(),
            writes: writes.into(),
            instruction: instruction.into(),
        };
        ctx.into()
    }
}

pub struct ModularMulDivCoreChip {
    pub air: ModularMulDivCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl ModularMulDivCoreChip {
    pub fn new(
        config: ExprBuilderConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = ModularMulDivCoreAir::new(config, range_checker.bus(), offset);
        Self { air, range_checker }
    }
}

pub struct ModularMulDivCoreRecord {
    pub x: BigUint,
    pub y: BigUint,
    pub is_mul_flag: bool,
    pub is_div_flag: bool,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for ModularMulDivCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = ModularMulDivCoreRecord;
    type Air = ModularMulDivCoreAir;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let num_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_idx = opcode - self.air.offset;
        let data: DynArray<_> = reads.into();
        let data = data.0;
        assert_eq!(data.len(), 2 * num_limbs);
        let x = data[..num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let y = data[num_limbs..]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();

        let x_biguint = limbs_to_biguint(&x, limb_bits);
        let y_biguint = limbs_to_biguint(&y, limb_bits);

        let local_opcode = Rv32ModularArithmeticOpcode::from_usize(local_opcode_idx);
        let is_mul_flag = match local_opcode {
            // for SETUP_MULDIV, we want to fictiously multiply by zero and not divide
            Rv32ModularArithmeticOpcode::MUL => true,
            Rv32ModularArithmeticOpcode::DIV | Rv32ModularArithmeticOpcode::SETUP_MULDIV => false,
            _ => panic!("Unsupported opcode: {:?}", local_opcode),
        };
        let is_div_flag = match local_opcode {
            Rv32ModularArithmeticOpcode::DIV => true,
            Rv32ModularArithmeticOpcode::MUL | Rv32ModularArithmeticOpcode::SETUP_MULDIV => false,
            _ => panic!("Unsupported opcode: {:?}", local_opcode),
        };

        let vars = self.air.expr.execute(
            vec![x_biguint.clone(), y_biguint.clone()],
            vec![is_mul_flag, is_div_flag],
        );
        assert_eq!(vars.len(), 1);
        let z_biguint = vars[0].clone();
        tracing::trace!(
            "ModularArithmeticOpcode | {local_opcode:?} | {z_biguint:?} | {x_biguint:?} | {y_biguint:?}",
        );
        let z_limbs = biguint_to_limbs_vec(z_biguint, limb_bits, num_limbs);
        let writes = z_limbs.into_iter().map(F::from_canonical_u32).collect_vec();
        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);

        Ok((
            ctx.into(),
            ModularMulDivCoreRecord {
                x: x_biguint,
                y: y_biguint,
                is_mul_flag,
                is_div_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "ModularMulDiv".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.air.expr.generate_subrow(
            (
                &self.range_checker,
                vec![record.x, record.y],
                vec![record.is_mul_flag, record.is_div_flag],
            ),
            row_slice,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
