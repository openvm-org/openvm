use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_primitives::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use ax_ecc_primitives::field_expression::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExprCols, FieldVariable,
};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::{instruction::Instruction, Rv32ModularArithmeticOpcode};
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::UsizeOpcode, AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface,
        DynArray, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};

/// The number of limbs and limb bits are determined at runtime.
#[derive(Clone)]
pub struct ModularAddSubCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl ModularAddSubCoreAir {
    pub fn new(
        config: ExprBuilderConfig,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
    ) -> Self {
        config.check_valid();
        let builder = ExprBuilder::new(config, range_bus.range_max_bits);
        let builder = Rc::new(RefCell::new(builder));
        let x1 = ExprBuilder::new_input(builder.clone());
        let x2 = ExprBuilder::new_input(builder.clone());
        let x3 = x1.clone() + x2.clone();
        let x4 = x1 - x2;
        let is_add_flag = builder.borrow_mut().new_flag();
        let mut x5 = FieldVariable::select(is_add_flag, &x3, &x4);
        x5.save();
        let builder = builder.borrow().clone();

        let expr = FieldExpr::new(builder, range_bus);
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for ModularAddSubCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for ModularAddSubCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for ModularAddSubCoreAir
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
            is_setup,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), 2);
        assert_eq!(vars.len(), 1);
        assert_eq!(flags.len(), 1);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[0].iter().map(|x| (*x).into()).collect();

        let local_opcode_idx = flags[0]
            * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::ADD as usize)
            + is_setup
                * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::SETUP as usize)
            + (AB::Expr::ONE - is_setup - flags[0])
                * AB::Expr::from_canonical_usize(Rv32ModularArithmeticOpcode::SUB as usize);

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

/// Number of limbs and limb size are determined purely at runtime
pub struct ModularAddSubCoreChip {
    pub air: ModularAddSubCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl ModularAddSubCoreChip {
    pub fn new(
        config: ExprBuilderConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = ModularAddSubCoreAir::new(config, range_checker.bus(), offset);
        Self { air, range_checker }
    }
}

pub struct ModularAddSubCoreRecord {
    pub x: BigUint,
    pub y: BigUint,
    pub is_add_flag: bool,
    pub is_setup: bool,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for ModularAddSubCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = ModularAddSubCoreRecord;
    type Air = ModularAddSubCoreAir;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let num_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_index = opcode - self.air.offset;
        let data: DynArray<_> = reads.into();
        let data = data.0;
        debug_assert_eq!(data.len(), 2 * num_limbs);
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

        let local_opcode = Rv32ModularArithmeticOpcode::from_usize(local_opcode_index);
        let is_add_flag = match local_opcode {
            Rv32ModularArithmeticOpcode::ADD => true,
            Rv32ModularArithmeticOpcode::SUB | Rv32ModularArithmeticOpcode::SETUP => false,
            _ => panic!("Unsupported opcode: {:?}", local_opcode),
        };
        let is_setup_flag = match local_opcode {
            Rv32ModularArithmeticOpcode::SETUP => true,
            Rv32ModularArithmeticOpcode::ADD | Rv32ModularArithmeticOpcode::SUB => false,
            _ => panic!("Unsupported opcode: {:?}", local_opcode),
        };

        let vars = self.air.expr.execute(
            vec![x_biguint.clone(), y_biguint.clone()],
            vec![is_add_flag],
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
            ModularAddSubCoreRecord {
                x: x_biguint,
                y: y_biguint,
                is_add_flag,
                is_setup: is_setup_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "ModularAddSub".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.air.expr.generate_subrow(
            (
                &self.range_checker,
                vec![record.x, record.y],
                vec![record.is_add_flag],
            ),
            row_slice,
        );
        row_slice[1] = F::from_bool(record.is_setup);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
