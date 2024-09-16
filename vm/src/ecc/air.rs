use std::{borrow::Borrow, mem::size_of};

use afs_primitives::{
    bigint::{CanonicalUint, DefaultLimbConfig},
    ecc::{
        EcAddIoCols as EcAddPrimitiveIoCols, EcAddUnequalAir, EcAuxCols as EcAddPrimitiveAuxCols,
        EcPoint,
    },
    sub_chip::SubAir,
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::columns::*;
use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode,
    },
    cpu::trace::Instruction,
    memory::{
        offline_checker::MemoryBridge, MemoryChipRef, MemoryHeapReadRecord, MemoryHeapWriteRecord,
    },
    modular_arithmetic::NUM_LIMBS,
};

#[derive(Clone, Debug)]
pub struct EcAddUnequalVmAir {
    pub air: EcAddUnequalAir,
    pub execution_bus: ExecutionBus,
    pub memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for EcAddUnequalVmAir {
    fn width(&self) -> usize {
        size_of::<EcAddUnequalCols<F>>()
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

        let aux = EcAddPrimitiveAuxCols {
            is_valid: cols.aux.aux.is_valid,
            lambda: cols.aux.aux.lambda.clone(),
            lambda_check: cols.aux.aux.lambda_check.clone(),
            x3_check: cols.aux.aux.x3_check.clone(),
            y3_check: cols.aux.aux.y3_check.clone(),
        };

        SubAir::eval(&self.air, builder, io, aux);

        // TODO: interactions.
    }
}
