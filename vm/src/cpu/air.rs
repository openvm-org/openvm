use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecIoCols, IsEqualVecAir},
    is_zero::{columns::IsZeroIoCols, IsZeroAir},
    sub_chip::SubAir,
};

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols},
    max_accesses_per_instruction, CpuAir,
    OpCode::*,
    CPU_MAX_READS_PER_CYCLE, FIELD_ARITHMETIC_INSTRUCTIONS, INST_WIDTH,
};

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for CpuAir<WORD_SIZE> {
    fn width(&self) -> usize {
        CpuCols::<WORD_SIZE, F>::get_width(self.options)
    }
}

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    fn assert_compose<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        word: [AB::Var; WORD_SIZE],
        field_elem: AB::Expr,
    ) {
        builder.assert_eq(word[0], field_elem);
        for &cell in word.iter().take(WORD_SIZE).skip(1) {
            builder.assert_zero(cell);
        }
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> Air<AB> for CpuAir<WORD_SIZE> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let inst_width = AB::F::from_canonical_usize(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CpuCols::<WORD_SIZE, AB::Var>::from_slice(local, self.options);

        let next = main.row_slice(1);
        let next: &[AB::Var] = (*next).borrow();
        let next_cols = CpuCols::<WORD_SIZE, AB::Var>::from_slice(next, self.options);
        let CpuCols { io, aux } = local_cols;
        let CpuCols { io: next_io, .. } = next_cols;

        let CpuIoCols {
            timestamp,
            pc,
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
        } = io;
        let CpuIoCols {
            timestamp: next_timestamp,
            pc: next_pc,
            ..
        } = next_io;

        let CpuAuxCols {
            operation_flags,
            accesses,
            read0_equals_read1,
            is_equal_vec_aux,
        } = aux;

        let read1 = &accesses[0];
        let read2 = &accesses[1];
        let write = &accesses[CPU_MAX_READS_PER_CYCLE];

        // set correct operation flag
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
        self.assert_compose(&mut when_loadw, read1.data, read2.address - b);

        when_loadw.assert_eq(write.address_space, d);
        when_loadw.assert_eq(write.address, a);
        for i in 0..WORD_SIZE {
            when_loadw.assert_eq(write.data[i], read2.data[i]);
        }

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
        self.assert_compose(&mut when_storew, read1.data, write.address - b);
        for i in 0..WORD_SIZE {
            when_storew.assert_eq(write.data[i], read2.data[i]);
        }

        when_storew
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // SHINTW: e[d[a] + b] <- ?
        let shintw_flag = operation_flags[&SHINTW];
        read1_enabled_check = read1_enabled_check + shintw_flag;
        write_enabled_check = write_enabled_check + shintw_flag;

        let mut when_shintw = builder.when(shintw_flag);
        when_shintw.assert_eq(read1.address_space, d);
        when_shintw.assert_eq(read1.address, a);

        when_shintw.assert_eq(write.address_space, e);
        self.assert_compose(&mut when_shintw, read1.data, write.address - b);

        when_shintw
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let jal_flag = operation_flags[&JAL];
        write_enabled_check = write_enabled_check + jal_flag;

        let mut when_jal = builder.when(jal_flag);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.address, a);
        self.assert_compose(&mut when_jal, write.data, pc + inst_width);

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
            .when(read0_equals_read1)
            .assert_eq(next_pc, pc + c);
        when_beq
            .when_transition()
            .when(AB::Expr::one() - read0_equals_read1)
            .assert_eq(next_pc, pc + inst_width);

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
            .when(read0_equals_read1)
            .assert_eq(next_pc, pc + inst_width);
        when_bne
            .when_transition()
            .when(AB::Expr::one() - read0_equals_read1)
            .assert_eq(next_pc, pc + c);

        // TERMINATE
        let terminate_flag = operation_flags[&TERMINATE];
        let mut when_terminate = builder.when(terminate_flag);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // arithmetic operations
        if self.options.field_arithmetic_enabled {
            let mut arithmetic_flags = AB::Expr::zero();
            for opcode in FIELD_ARITHMETIC_INSTRUCTIONS {
                arithmetic_flags += operation_flags[&opcode].into();
            }
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
            let is_zero_io = IsZeroIoCols {
                x: access.address_space,
                is_zero: access.is_immediate,
            };
            let is_zero_aux = access.is_zero_aux;
            SubAir::eval(&IsZeroAir, builder, is_zero_io, is_zero_aux);
        }
        for read in [&read1, &read2] {
            self.assert_compose(
                &mut builder.when(read.is_immediate),
                read.data,
                read.address.into(),
            );
        }
        // maybe writes to immediate address space are ignored instead of disallowed?
        //builder.assert_zero(write.is_immediate);

        // evaluate equality between read1 and read2

        let is_equal_vec_io_cols = IsEqualVecIoCols {
            x: read1.data.to_vec(),
            y: read2.data.to_vec(),
            is_equal: read0_equals_read1,
        };
        SubAir::eval(
            &IsEqualVecAir::new(WORD_SIZE),
            builder,
            is_equal_vec_io_cols,
            is_equal_vec_aux,
        );

        // make sure program starts at beginning
        builder.when_first_row().assert_zero(pc);
        builder.when_first_row().assert_zero(timestamp);

        // update the timestamp correctly
        for (&opcode, &flag) in operation_flags.iter() {
            if opcode != TERMINATE {
                builder.when(flag).assert_eq(
                    next_timestamp,
                    timestamp + AB::F::from_canonical_usize(max_accesses_per_instruction(opcode)),
                )
            }
        }

        // make sure program terminates
        builder
            .when_last_row()
            .assert_eq(opcode, AB::Expr::from_canonical_usize(TERMINATE as usize));

        // check accesses enabled
        builder.assert_eq(read1.enabled, read1_enabled_check);
        builder.assert_eq(read2.enabled, read2_enabled_check);
        builder.assert_eq(write.enabled, write_enabled_check);

        // Turn on all interactions
        self.eval_interactions(builder, io, accesses, &operation_flags);
    }
}
