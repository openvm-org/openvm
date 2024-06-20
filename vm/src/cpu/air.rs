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

        let inst_width = AB::Var::from_canonical_usize(INST_WIDTH);

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

        for &operation_flag in operation_flags.iter() {
            builder.assert_bool(operation_flag);
        }

        let mut sum_flags = AB::Expr::zero();
        let mut match_opcode = AB::Expr::zero();
        for (i, flag) in operation_flags.iter().enumerate() {
            sum_flags = sum_flags + *flag;
            match_opcode += *flag * AB::Expr::from_canonical_u64(i.try_into().unwrap());
        }
        builder.assert_one(sum_flags);
        builder.assert_eq(opcode, match_opcode);

        // LOADW: d[a] <- e[d[c] + b]
        let mut when_loadw = builder.when(operation_flags[LOADW as usize]);

        when_loadw.assert_eq(read1.address_space, d);
        when_loadw.assert_eq(read1.address, c);

        when_loadw.assert_eq(read2.address_space, e);
        when_loadw.assert_eq(read2.address, read1.data + b);

        when_loadw.assert_eq(write.address_space, d);
        when_loadw.assert_eq(write.address, a);
        when_loadw.assert_eq(write.data, read2.data);

        when_loadw.when_transition()
            .assert_eq(next_pc, pc + inst_width.clone());

        // STOREW: e[d[c] + b] <- d[a]
        let mut when_storew = builder.when(operation_flags[STOREW as usize]);
        when_storew.assert_eq(read1.address_space, d);
        when_storew.assert_eq(read1.address, c);

        when_storew.assert_eq(read2.address_space, d);
        when_storew.assert_eq(read2.address, a);

        when_storew.assert_eq(write.address_space, e);
        when_storew.assert_eq(write.address, read1.data + b);
        when_storew.assert_eq(write.data, read2.data);

        when_storew.when_transition()
            .assert_eq(next_pc, pc + inst_width.clone());

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let mut when_jal = builder.when(operation_flags[JAL as usize]);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.address, a);
        when_jal.assert_eq(
            write.data,
            pc + AB::Expr::from_canonical_u64(INST_WIDTH.try_into().unwrap()),
        );

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let mut when_beq = builder.when(operation_flags[BEQ as usize]);

        when_beq.assert_eq(read1.address_space, d);
        when_beq.assert_eq(read1.address, a);

        when_beq.assert_eq(read2.address_space, e);
        when_beq.assert_eq(read2.address, b);

        when_beq.when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + c);
        when_beq.when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + inst_width.clone());

        let is_equal_io_cols = IsEqualIOCols {
            x: read1.data,
            y: read2.data,
            is_equal: beq_check,
        };
        let is_equal_aux_cols = IsEqualAuxCols { inv: is_equal_aux };
        SubAir::eval(&IsEqualAir, builder, is_equal_io_cols, is_equal_aux_cols);

        // BNE: If d[a] != e[b], pc <- pc + c
        let mut when_bne = builder.when(operation_flags[BNE as usize]);

        when_bne.assert_eq(read1.address_space, d);
        when_bne.assert_eq(read1.address, a);

        when_bne.assert_eq(read2.address_space, e);
        when_bne.assert_eq(read2.address, b);

        when_bne.when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + inst_width.clone());
        when_bne.when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + c);

        // TERMINATE
        let mut when_terminate = builder.when(operation_flags[TERMINATE as usize]);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // arithmetic operations
        if self.options.field_arithmetic_enabled {
            let mut when_arithmetic = builder.when(
                operation_flags[FADD as usize]
                    + operation_flags[FSUB as usize]
                    + operation_flags[FMUL as usize]
                    + operation_flags[FDIV as usize],
            );

            // read from e[b] and e[c]
            when_arithmetic.assert_eq(read1.address_space, e);
            when_arithmetic.assert_eq(read1.address, b);

            when_arithmetic.assert_eq(read2.address_space, e);
            when_arithmetic.assert_eq(read2.address, c);

            // write to d[a]
            when_arithmetic.assert_eq(write.address_space, d);
            when_arithmetic.assert_eq(write.address, a);

            when_arithmetic.when_transition()
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
                .assert_eq(read.data, read.address);
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

        // make sure program terminates
        builder.when_last_row().assert_eq(opcode, AB::Expr::from_canonical_usize(TERMINATE as usize));
    }
}
