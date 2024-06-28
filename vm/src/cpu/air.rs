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
    columns::{CpuAuxCols, CpuCols, CpuIoCols},
    CpuAir,
    OpCode::*,
    INST_WIDTH,
};

impl<F: Field> BaseAir<F> for CpuAir {
    fn width(&self) -> usize {
        CpuCols::<F>::get_width(self.options)
    }
}

impl<AB: AirBuilder> Air<AB> for CpuAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let inst_width = AB::F::from_canonical_usize(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CpuCols::<AB::Var>::from_slice(local, self.options);

        let next = main.row_slice(1);
        let next: &[AB::Var] = (*next).borrow();
        let next_cols = CpuCols::<AB::Var>::from_slice(next, self.options);
        let CpuCols { io, aux } = local_cols;
        let CpuCols { io: next_io, .. } = next_cols;

        let CpuIoCols {
            clock_cycle: clock,
            pc,
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
        } = io;
        let CpuIoCols {
            clock_cycle: next_clock,
            pc: next_pc,
            ..
        } = next_io;

        let CpuAuxCols {
            operation_flags,
            read1,
            read2,
            write,
            beq_check,
            is_equal_aux,
        } = aux;

        for &flag in operation_flags.values() {
            builder.assert_bool(flag);
        }

        let mut sum_flags = AB::Expr::zero();
        let mut match_opcode = AB::Expr::zero();
        for (&opcode, &flag) in operation_flags.iter() {
            sum_flags = sum_flags + flag;
            match_opcode += flag * AB::F::from_canonical_usize(opcode as usize);
        }
        builder.assert_one(sum_flags);
        builder.assert_eq(opcode, match_opcode);

        // keep track of when memory accesses should be enabled
        let mut read1_enabled_check = AB::Expr::zero();
        let mut read2_enabled_check = AB::Expr::zero();
        let mut write_enabled_check = AB::Expr::zero();

        // LOADW: d[a] <- e[d[c] + b]
        let loadw_flag = operation_flags[&LOADW];
        read1_enabled_check = read1_enabled_check + loadw_flag;
        read2_enabled_check = read2_enabled_check + loadw_flag;
        write_enabled_check = write_enabled_check + loadw_flag;

        let mut when_loadw = builder.when(loadw_flag);

        when_loadw.assert_eq(read1.address_space, d);
        when_loadw.assert_eq(read1.address, c);

        when_loadw.assert_eq(read2.address_space, e);
        when_loadw.assert_eq(read2.address, read1.data + b);

        when_loadw.assert_eq(write.address_space, d);
        when_loadw.assert_eq(write.address, a);
        when_loadw.assert_eq(write.data, read2.data);

        when_loadw
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // STOREW: e[d[c] + b] <- d[a]
        let storew_flag = operation_flags[&STOREW];
        read1_enabled_check = read1_enabled_check + storew_flag;
        read2_enabled_check = read2_enabled_check + storew_flag;
        write_enabled_check = write_enabled_check + storew_flag;

        let mut when_storew = builder.when(storew_flag);
        when_storew.assert_eq(read1.address_space, d);
        when_storew.assert_eq(read1.address, c);

        when_storew.assert_eq(read2.address_space, d);
        when_storew.assert_eq(read2.address, a);

        when_storew.assert_eq(write.address_space, e);
        when_storew.assert_eq(write.address, read1.data + b);
        when_storew.assert_eq(write.data, read2.data);

        when_storew
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let jal_flag = operation_flags[&JAL];
        write_enabled_check = write_enabled_check + jal_flag;

        let mut when_jal = builder.when(jal_flag);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.address, a);
        when_jal.assert_eq(write.data, pc + inst_width);

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let beq_flag = operation_flags[&BEQ];
        read1_enabled_check = read1_enabled_check + beq_flag;
        read2_enabled_check = read2_enabled_check + beq_flag;

        let mut when_beq = builder.when(beq_flag);

        when_beq.assert_eq(read1.address_space, d);
        when_beq.assert_eq(read1.address, a);

        when_beq.assert_eq(read2.address_space, e);
        when_beq.assert_eq(read2.address, b);

        when_beq
            .when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + c);
        when_beq
            .when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + inst_width);

        let is_equal_io_cols = IsEqualIOCols {
            x: read1.data,
            y: read2.data,
            is_equal: beq_check,
        };
        let is_equal_aux_cols = IsEqualAuxCols { inv: is_equal_aux };
        SubAir::eval(&IsEqualAir, builder, is_equal_io_cols, is_equal_aux_cols);

        // BNE: If d[a] != e[b], pc <- pc + c
        let bne_flag = operation_flags[&BNE];
        read1_enabled_check = read1_enabled_check + bne_flag;
        read2_enabled_check = read2_enabled_check + bne_flag;

        let mut when_bne = builder.when(bne_flag);

        when_bne.assert_eq(read1.address_space, d);
        when_bne.assert_eq(read1.address, a);

        when_bne.assert_eq(read2.address_space, e);
        when_bne.assert_eq(read2.address, b);

        when_bne
            .when_transition()
            .when(beq_check)
            .assert_eq(next_pc, pc + inst_width);
        when_bne
            .when_transition()
            .when(AB::Expr::one() - beq_check)
            .assert_eq(next_pc, pc + c);

        // TERMINATE
        let terminate_flag = operation_flags[&TERMINATE];
        let mut when_terminate = builder.when(terminate_flag);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // arithmetic operations
        if self.options.field_arithmetic_enabled {
            let arithmetic_flags = operation_flags[&FADD]
                + operation_flags[&FSUB]
                + operation_flags[&FMUL]
                + operation_flags[&FDIV];
            read1_enabled_check += arithmetic_flags.clone();
            read2_enabled_check += arithmetic_flags.clone();
            write_enabled_check += arithmetic_flags.clone();
            let mut when_arithmetic = builder.when(arithmetic_flags);

            // read from d[b] and e[c]
            when_arithmetic.assert_eq(read1.address_space, d);
            when_arithmetic.assert_eq(read1.address, b);

            when_arithmetic.assert_eq(read2.address_space, e);
            when_arithmetic.assert_eq(read2.address, c);

            // write to d[a]
            when_arithmetic.assert_eq(write.address_space, d);
            when_arithmetic.assert_eq(write.address, a);

            when_arithmetic
                .when_transition()
                .assert_eq(next_pc, pc + inst_width);
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
        builder
            .when_last_row()
            .assert_eq(opcode, AB::Expr::from_canonical_usize(TERMINATE as usize));

        // check accesses enabled
        builder.assert_eq(read1.enabled, read1_enabled_check);
        builder.assert_eq(read2.enabled, read2_enabled_check);
        builder.assert_eq(write.enabled, write_enabled_check);
    }
}
