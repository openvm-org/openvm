use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use afs_chips::{
    is_equal::{
        columns::{IsEqualAuxCols, IsEqualIOCols},
        IsEqualAir,
    },
    is_zero::{columns::IsZeroIOCols, IsZeroAir},
    sub_chip::SubAir,
};

use super::{
    columns::{CPUAuxCols, CPUCols, CPUIOCols},
    CPUAir,
    OpCode::*,
    INST_WIDTH,
};

impl<F: Field> BaseAir<F> for CPUAir {
    fn width(&self) -> usize {
        CPUCols::<F>::get_width(self.options)
    }
}

impl<AB: AirBuilder> Air<AB> for CPUAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let inst_width = AB::Expr::from_canonical_u64(INST_WIDTH.try_into().unwrap());

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CPUCols::<AB::Var>::from_slice(local, self.options);

        let next = main.row_slice(1);
        let next: &[AB::Var] = (*next).borrow();
        let next_cols = CPUCols::<AB::Var>::from_slice(next, self.options);
        let CPUCols { io, aux } = local_cols;
        let CPUCols { io: next_io, .. } = next_cols;

        let CPUIOCols {
            clock_cycle: clock,
            pc,
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            as_b: d,
            as_c: e,
        } = io;
        let CPUIOCols {
            clock_cycle: next_clock,
            pc: next_pc,
            ..
        } = next_io;

        let CPUAuxCols {
            operation_flags,
            read1,
            read2,
            write,
            beq_check,
            is_equal_aux,
        } = aux;
        // set correct operation flag

        for operation_flag in operation_flags.iter() {
            builder.assert_bool(*operation_flag);
        }

        let mut sum_flags = AB::Expr::zero();
        let mut match_opcode = AB::Expr::zero();
        for (i, flag) in operation_flags.iter().enumerate() {
            sum_flags = sum_flags + *flag;
            match_opcode += *flag * AB::Expr::from_canonical_u64(i.try_into().unwrap());
        }
        builder.assert_one(sum_flags);
        builder.assert_eq(opcode, match_opcode);

        // LOADF: d[a] <- e[d[c] + b]
        let mut here = builder.when(operation_flags[LOADW as usize]);

        here.assert_eq(read1.address_space, d);
        here.assert_eq(read1.address, c);

        here.assert_eq(read2.address_space, e);
        here.assert_eq(read2.address, read1.value + b);

        here.assert_eq(write.address_space, d);
        here.assert_eq(write.address, a);
        here.assert_eq(write.value, read2.value);

        here.when_transition()
            .assert_eq(next_pc, pc + inst_width.clone());

        // STOREF: e[d[c] + b] <- d[a]
        let mut here = builder.when(operation_flags[STOREW as usize]);
        here.assert_eq(read1.address_space, d);
        here.assert_eq(read1.address, c);

        here.assert_eq(read2.address_space, d);
        here.assert_eq(read2.address, a);

        here.assert_eq(write.address_space, e);
        here.assert_eq(write.address, read1.value + b);
        here.assert_eq(write.value, read2.value);

        here.when_transition()
            .assert_eq(next_pc, pc + inst_width.clone());

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let mut here = builder.when(operation_flags[JAL as usize]);

        here.assert_eq(write.address_space, d);
        here.assert_eq(write.address, a);
        here.assert_eq(
            write.value,
            pc + AB::Expr::from_canonical_u64(INST_WIDTH.try_into().unwrap()),
        );

        here.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let mut here = builder.when(operation_flags[BEQ as usize]);

        here.assert_eq(read1.address_space, d);
        here.assert_eq(read1.address, a);

        here.assert_eq(read2.address_space, e);
        here.assert_eq(read2.address, b);

        here.when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + c);
        here.when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + inst_width.clone());

        let is_equal_io_cols = IsEqualIOCols {
            x: read1.value,
            y: read2.value,
            is_equal: beq_check,
        };
        let is_equal_aux_cols = IsEqualAuxCols { inv: is_equal_aux };
        SubAir::eval(&IsEqualAir, builder, is_equal_io_cols, is_equal_aux_cols);

        // BNE: If d[a] != e[b], pc <- pc + c
        let mut here = builder.when(operation_flags[BNE as usize]);

        here.assert_eq(read1.address_space, d);
        here.assert_eq(read1.address, a);

        here.assert_eq(read2.address_space, e);
        here.assert_eq(read2.address, b);

        here.when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + inst_width.clone());
        here.when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + c);

        // arithmetic operations
        if self.options.field_arithmetic_enabled {
            let mut here = builder.when(
                operation_flags[FADD as usize]
                    + operation_flags[FSUB as usize]
                    + operation_flags[FMUL as usize]
                    + operation_flags[FDIV as usize],
            );

            // read from e[b] and e[c]
            here.assert_eq(read1.address_space, e);
            here.assert_eq(read1.address, b);

            here.assert_eq(read2.address_space, e);
            here.assert_eq(read2.address, c);

            // write to d[a]
            here.assert_eq(write.address_space, d);
            here.assert_eq(write.address, a);

            here.when_transition()
                .assert_eq(next_pc, pc + inst_width.clone());
        }

        // immediate calculation

        for access in [&read1, &read2, &write] {
            let is_zero_io = IsZeroIOCols {
                x: access.address_space,
                is_zero: access.is_immediate,
            };
            let is_zero_aux = access.is_zero_aux;
            SubAir::eval(&IsZeroAir, builder, is_zero_io, is_zero_aux);
        }
        for read in [&read1, &read2] {
            builder
                .when(read.is_immediate)
                .assert_eq(read.value, read.address);
        }
        // maybe writes to immediate address space are ignored instead of disallowed?
        //builder.assert_zero(write.is_immediate);

        // make sure program starts at beginning
        builder.when_first_row().assert_zero(pc);
        builder.when_first_row().assert_zero(clock);

        // make sure time works like it usually does
        builder
            .when_transition()
            .assert_eq(next_clock, clock + AB::Expr::one());

        // termination?
        // ???
    }
}
