use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::{
    arch::columns::{ExecutionState, InstructionCols},
    cpu::{MEMORY_BUS, WORD_SIZE},
};

use super::{columns::FieldExtensionArithmeticCols, EXTENSION_DEGREE, FieldExtensionArithmeticAir};

fn eval_rw_interactions<AB: InteractionBuilder>(
    builder: &mut AB,
    is_write: bool,
    local: &FieldExtensionArithmeticCols<AB::Var>,
    addr_space: AB::Var,
    address: AB::Var,
    ext_element_ind: usize,
) {
    let io = &local.io;
    let aux = &local.aux;

    let ext_element = if ext_element_ind == 0 {
        io.x
    } else if ext_element_ind == 1 {
        io.y
    } else {
        io.z
    };

    for (i, element) in ext_element.into_iter().enumerate() {
        let timestamp = aux.start_timestamp + AB::F::from_canonical_usize(ext_element_ind * 4 + i);

        let pointer = address + AB::F::from_canonical_usize(i * WORD_SIZE);

        let fields = [
            timestamp,
            AB::Expr::from_bool(is_write),
            addr_space.into(),
            pointer,
        ]
        .into_iter() // TODO: Handle WORD_SIZE > 1 later
        .chain([element.into()]);

        if ext_element_ind == 1 {
            builder.push_send(MEMORY_BUS, fields, aux.valid_y_read);
        } else {
            builder.push_send(MEMORY_BUS, fields, aux.is_valid);
        }
    }
}

impl FieldExtensionArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &FieldExtensionArithmeticCols<AB::Var>,
    ) {
        // reads for x
        eval_rw_interactions(builder, false, local, local.aux.d, local.aux.op_b, 0);
        // reads for y
        eval_rw_interactions(builder, false, local, local.aux.e, local.aux.op_c, 1);
        // writes for z
        eval_rw_interactions(builder, true, local, local.aux.d, local.aux.op_a, 2);

        self.execution_bus.interact_execute(
            builder,
            ExecutionState {
                pc: local.aux.pc.into(),
                timestamp: local.aux.start_timestamp.into(),
            },
            ExecutionState {
                pc: local.aux.pc + AB::F::one(),
                timestamp: local.aux.start_timestamp
                    + AB::F::from_canonical_usize(3 * EXTENSION_DEGREE),
            },
            InstructionCols {
                opcode: local.io.opcode.into(),
                a: local.aux.op_a.into(),
                b: local.aux.op_b.into(),
                c: local.aux.op_c.into(),
                d: local.aux.d.into(),
                e: local.aux.e.into(),
                f: AB::Expr::zero(),
                g: AB::Expr::zero(),
            },
        );
    }
}
