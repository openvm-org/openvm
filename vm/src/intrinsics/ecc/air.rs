use std::borrow::Borrow;

use afs_primitives::{
    bigint::{CanonicalUint, DefaultLimbConfig},
    ecc::{
        EcAddIoCols as EcAddPrimitiveIoCols, EcAddUnequalAir, EcAuxCols as EcPrimitiveAuxCols,
        EcDoubleAir, EcDoubleIoCols as EcDoublePrimitiveIoCols, EcPoint,
    },
    sub_chip::SubAir,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::*, NUM_LIMBS};
use crate::{arch::ExecutionBridge, system::memory::offline_checker::MemoryBridge};

#[derive(Clone, Debug)]
pub struct EcAddUnequalVmAir {
    pub air: EcAddUnequalAir,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,

    pub(super) offset: usize,
}

impl<F: Field> BaseAirWithPublicValues<F> for EcAddUnequalVmAir {}
impl<F: Field> PartitionedBaseAir<F> for EcAddUnequalVmAir {}
impl<F: Field> BaseAir<F> for EcAddUnequalVmAir {
    fn width(&self) -> usize {
        EcAddUnequalCols::<F>::width(&self.air.config)
    }
}

impl<AB: InteractionBuilder> Air<AB> for EcAddUnequalVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let cols: &[AB::Var] = (*local).borrow();
        let cols =
            EcAddUnequalCols::<AB::Var>::from_iterator(cols.iter().copied(), &self.air.config);

        let p1 = EcPoint {
            x: CanonicalUint::<AB::Var, DefaultLimbConfig>::from_vec(
                cols.io.p1.data.data[..NUM_LIMBS].to_vec(),
            ),
            y: CanonicalUint::from_vec(cols.io.p1.data.data[NUM_LIMBS..].to_vec()),
        };
        let p2 = EcPoint {
            x: CanonicalUint::from_vec(cols.io.p2.data.data[..NUM_LIMBS].to_vec()),
            y: CanonicalUint::from_vec(cols.io.p2.data.data[NUM_LIMBS..].to_vec()),
        };
        let p3 = EcPoint {
            x: CanonicalUint::from_vec(cols.io.p3.data.data[..NUM_LIMBS].to_vec()),
            y: CanonicalUint::from_vec(cols.io.p3.data.data[NUM_LIMBS..].to_vec()),
        };
        let io = EcAddPrimitiveIoCols { p1, p2, p3 };

        let aux = EcPrimitiveAuxCols {
            is_valid: cols.aux.aux.is_valid,
            lambda: cols.aux.aux.lambda.clone(),
            lambda_check: cols.aux.aux.lambda_check.clone(),
            x3_check: cols.aux.aux.x3_check.clone(),
            y3_check: cols.aux.aux.y3_check.clone(),
        };

        SubAir::eval(&self.air, builder, io, aux);

        self.eval_interactions(builder, cols.io, cols.aux);
    }
}

#[derive(Clone, Debug)]
pub struct EcDoubleVmAir {
    pub air: EcDoubleAir,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,

    pub(super) offset: usize,
}

impl<F: Field> BaseAirWithPublicValues<F> for EcDoubleVmAir {}
impl<F: Field> PartitionedBaseAir<F> for EcDoubleVmAir {}
impl<F: Field> BaseAir<F> for EcDoubleVmAir {
    fn width(&self) -> usize {
        EcDoubleCols::<F>::width(&self.air.config)
    }
}

impl<AB: InteractionBuilder> Air<AB> for EcDoubleVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let cols: &[AB::Var] = (*local).borrow();
        let cols = EcDoubleCols::<AB::Var>::from_iterator(cols.iter().copied(), &self.air.config);

        let p1 = EcPoint {
            x: CanonicalUint::<AB::Var, DefaultLimbConfig>::from_vec(
                cols.io.p1.data.data[..NUM_LIMBS].to_vec(),
            ),
            y: CanonicalUint::from_vec(cols.io.p1.data.data[NUM_LIMBS..].to_vec()),
        };
        let p2 = EcPoint {
            x: CanonicalUint::from_vec(cols.io.p2.data.data[..NUM_LIMBS].to_vec()),
            y: CanonicalUint::from_vec(cols.io.p2.data.data[NUM_LIMBS..].to_vec()),
        };
        let io = EcDoublePrimitiveIoCols { p1, p2 };

        let aux = EcPrimitiveAuxCols {
            is_valid: cols.aux.aux.is_valid,
            lambda: cols.aux.aux.lambda.clone(),
            lambda_check: cols.aux.aux.lambda_check.clone(),
            x3_check: cols.aux.aux.x3_check.clone(),
            y3_check: cols.aux.aux.y3_check.clone(),
        };

        SubAir::eval(&self.air, builder, io, aux);

        self.eval_interactions(builder, cols.io, cols.aux);
    }
}
