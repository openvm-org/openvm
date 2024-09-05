use std::{borrow::Borrow, iter::zip};

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecIoCols, IsEqualVecAir},
    sub_chip::SubAir,
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols},
    timestamp_delta, CpuOptions, INST_WIDTH, WORD_SIZE,
};
use crate::{
    arch::{bus::ExecutionBus, instructions::Opcode::*},
    memory::{
        offline_checker::bridge::{MemoryBridge, MemoryOfflineChecker},
        MemoryAddress,
    },
};

/// Air for the CPU. Carries no state and does not own execution.
#[derive(Clone, Debug)]
pub struct CpuAir {
    pub options: CpuOptions,
    pub execution_bus: ExecutionBus,
    pub memory_offline_checker: MemoryOfflineChecker,
}

impl<F: Field> BaseAir<F> for CpuAir {
    fn width(&self) -> usize {
        CpuCols::<F>::get_width(self)
    }
}

// TODO[osama]: here, there should be some relation enforced between the timestamp for the cpu and the memory timestamp
// TODO[osama]: also, rename to clk
impl<AB: AirBuilderWithPublicValues + InteractionBuilder> Air<AB> for CpuAir {
    // TODO: continuation verification checks program counters match up [INT-1732]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let start_pc = pis[0];
        let end_pc = pis[1];

        let inst_width = AB::F::from_canonical_usize(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CpuCols::from_slice(local, self);

        let next = main.row_slice(1);
        let next: &[AB::Var] = (*next).borrow();
        let next_cols = CpuCols::from_slice(next, self);
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
            op_f: f,
            op_g: g,
        } = io;
        let CpuIoCols {
            timestamp: next_timestamp,
            pc: next_pc,
            ..
        } = next_io;

        let CpuAuxCols {
            operation_flags,
            public_value_flags,
            reads,
            writes,
            read0_equals_read1,
            is_equal_vec_aux,
            reads_aux_cols,
            writes_aux_cols,
        } = aux;

        let [read1, read2, read3] = &reads;
        let [write] = &writes;

        // assert that the start pc is correct
        builder.when_first_row().assert_eq(pc, start_pc);
        builder.when_last_row().assert_eq(pc, end_pc);

        // set correct operation flag
        for &flag in operation_flags.values() {
            builder.assert_bool(flag);
        }

        let mut is_cpu_opcode = AB::Expr::zero();
        let mut match_opcode = AB::Expr::zero();
        for (&opcode, &flag) in operation_flags.iter() {
            is_cpu_opcode += flag.into();
            match_opcode += flag * AB::F::from_canonical_usize(opcode as usize);
        }
        builder.assert_bool(is_cpu_opcode.clone());
        builder
            .when(is_cpu_opcode.clone())
            .assert_eq(opcode, match_opcode);

        // keep track of when memory accesses should be enabled
        let mut read1_enabled_check = AB::Expr::zero();
        let mut read2_enabled_check = AB::Expr::zero();
        let mut read3_enabled_check = AB::Expr::zero();
        let mut write_enabled_check = AB::Expr::zero();

        // LOADW: d[a] <- e[d[c] + b + d[f] * g]
        let loadw_flag = operation_flags[&LOADW];
        read1_enabled_check = read1_enabled_check + loadw_flag;
        read2_enabled_check = read2_enabled_check + loadw_flag;
        write_enabled_check = write_enabled_check + loadw_flag;

        let mut when_loadw = builder.when(loadw_flag);

        when_loadw.assert_eq(read1.address_space, d);
        when_loadw.assert_eq(read1.pointer, c);

        when_loadw.assert_eq(read2.address_space, e);
        when_loadw.assert_eq(read1.value, read2.pointer - b);

        when_loadw.assert_eq(write.address_space, d);
        when_loadw.assert_eq(write.pointer, a);
        when_loadw.assert_eq(write.value, read2.value);

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
        when_storew.assert_eq(read1.pointer, c);

        when_storew.assert_eq(read2.address_space, d);
        when_storew.assert_eq(read2.pointer, a);

        when_storew.assert_eq(write.address_space, e);
        when_storew.assert_eq(read1.value, write.pointer - b);
        when_storew.assert_eq(write.value, read2.value);

        when_storew
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // LOADW2: d[a] <- e[d[c] + b + mem[f] * g]
        let loadw2_flag = operation_flags[&LOADW2];
        read1_enabled_check = read1_enabled_check + loadw2_flag;
        read2_enabled_check = read2_enabled_check + loadw2_flag;
        read3_enabled_check = read3_enabled_check + loadw2_flag;
        write_enabled_check = write_enabled_check + loadw2_flag;

        let mut when_loadw2 = builder.when(loadw2_flag);

        when_loadw2.assert_eq(read1.address_space, d);
        when_loadw2.assert_eq(read1.pointer, c);

        when_loadw2.assert_eq(read2.address_space, d);
        when_loadw2.assert_eq(read2.pointer, f);

        when_loadw2.assert_eq(read3.address_space, e);
        let addr_diff = read1.value + g * read2.value;
        when_loadw2.assert_eq(addr_diff, read3.pointer - b);

        when_loadw2.assert_eq(write.address_space, d);
        when_loadw2.assert_eq(write.pointer, a);
        when_loadw2.assert_eq(write.value, read3.value);

        when_loadw2
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // STOREW2: e[d[c] + b + mem[f] * g] <- d[a]
        let storew2_flag = operation_flags[&STOREW2];
        read1_enabled_check = read1_enabled_check + storew2_flag;
        read2_enabled_check = read2_enabled_check + storew2_flag;
        read3_enabled_check = read3_enabled_check + storew2_flag;
        write_enabled_check = write_enabled_check + storew2_flag;

        let mut when_storew2 = builder.when(storew2_flag);
        when_storew2.assert_eq(read1.address_space, d);
        when_storew2.assert_eq(read1.pointer, c);

        when_storew2.assert_eq(read2.address_space, d);
        when_storew2.assert_eq(read2.pointer, a);

        when_storew2.assert_eq(read3.address_space, d);
        when_storew2.assert_eq(read3.pointer, f);

        when_storew2.assert_eq(write.address_space, e);
        let addr_diff = read1.value + g * read3.value;
        when_storew2.assert_eq(addr_diff, write.pointer - b);
        when_storew2.assert_eq(write.value, read2.value);

        when_storew2
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // SHINTW: e[d[a] + b] <- ?
        let shintw_flag = operation_flags[&SHINTW];
        read1_enabled_check = read1_enabled_check + shintw_flag;
        write_enabled_check = write_enabled_check + shintw_flag;

        let mut when_shintw = builder.when(shintw_flag);
        when_shintw.assert_eq(read1.address_space, d);
        when_shintw.assert_eq(read1.pointer, a);

        when_shintw.assert_eq(write.address_space, e);
        when_shintw.assert_eq(read1.value, write.pointer - b);

        when_shintw
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let jal_flag = operation_flags[&JAL];
        write_enabled_check = write_enabled_check + jal_flag;

        let mut when_jal = builder.when(jal_flag);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.pointer, a);
        when_jal.assert_eq(write.value, pc + inst_width);

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let beq_flag = operation_flags[&BEQ];
        read1_enabled_check = read1_enabled_check + beq_flag;
        read2_enabled_check = read2_enabled_check + beq_flag;

        let mut when_beq = builder.when(beq_flag);

        when_beq.assert_eq(read1.address_space, d);
        when_beq.assert_eq(read1.pointer, a);

        when_beq.assert_eq(read2.address_space, e);
        when_beq.assert_eq(read2.pointer, b);

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
        when_bne.assert_eq(read1.pointer, a);

        when_bne.assert_eq(read2.address_space, e);
        when_bne.assert_eq(read2.pointer, b);

        when_bne
            .when_transition()
            .when(read0_equals_read1)
            .assert_eq(next_pc, pc + inst_width);
        when_bne
            .when_transition()
            .when(AB::Expr::one() - read0_equals_read1)
            .assert_eq(next_pc, pc + c);

        // NOP constraints same pc and timestamp as next row
        let nop_flag = operation_flags[&NOP];
        let mut when_nop = builder.when(nop_flag);
        when_nop.when_transition().assert_eq(next_pc, pc);
        when_nop
            .when_transition()
            .assert_eq(next_timestamp, timestamp);

        // TERMINATE
        let terminate_flag = operation_flags[&TERMINATE];
        let mut when_terminate = builder.when(terminate_flag);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // PUBLISH

        let publish_flag = operation_flags[&PUBLISH];
        read1_enabled_check = read1_enabled_check + publish_flag;
        read2_enabled_check = read2_enabled_check + publish_flag;

        let mut sum_flags = AB::Expr::zero();
        let mut match_public_value_index = AB::Expr::zero();
        let mut match_public_value = AB::Expr::zero();
        for (i, &flag) in public_value_flags.iter().enumerate() {
            builder.assert_bool(flag);
            sum_flags = sum_flags + flag;
            match_public_value_index += flag * AB::F::from_canonical_usize(i);
            match_public_value += flag * builder.public_values()[i + 2].into();
        }

        let mut when_publish = builder.when(publish_flag);

        when_publish.assert_one(sum_flags);
        when_publish.assert_eq(read1.value, match_public_value_index);
        when_publish.assert_eq(read2.value, match_public_value);

        when_publish.assert_eq(read1.address_space, d);
        when_publish.assert_eq(read1.pointer, a);

        when_publish.assert_eq(read2.address_space, e);
        when_publish.assert_eq(read2.pointer, b);

        // FIXME[zach]: Properly constrain op.enabled based on opcode.

        let mut op_timestamp: AB::Expr = io.timestamp.into();
        let memory_bridge = MemoryBridge::new(self.memory_offline_checker);
        for (read, read_aux_cols) in zip(&reads, reads_aux_cols) {
            memory_bridge
                .read_or_immediate(
                    MemoryAddress::new(read.address_space, read.pointer),
                    read.value,
                    op_timestamp.clone(),
                    read_aux_cols,
                )
                .eval(builder, read.enabled);
            op_timestamp += read.enabled.into();
        }

        for (write, write_aux_cols) in zip(&writes, writes_aux_cols) {
            memory_bridge
                .write(
                    MemoryAddress::new(write.address_space, write.pointer),
                    [write.value],
                    op_timestamp.clone(),
                    write_aux_cols,
                )
                .eval(builder, write.enabled);
            op_timestamp += write.enabled.into();
        }

        // evaluate equality between read1 and read2

        let is_equal_vec_io_cols = IsEqualVecIoCols {
            x: vec![read1.value],
            y: vec![read2.value],
            is_equal: read0_equals_read1,
        };
        SubAir::eval(
            &IsEqualVecAir::new(WORD_SIZE),
            builder,
            is_equal_vec_io_cols,
            is_equal_vec_aux,
        );

        // update the timestamp correctly
        for (&opcode, &flag) in operation_flags.iter() {
            if opcode != TERMINATE && opcode != NOP {
                builder.when(flag).assert_eq(
                    next_timestamp,
                    timestamp + AB::F::from_canonical_usize(timestamp_delta(opcode)),
                )
            }
        }

        // make sure program terminates or shards with NOP
        builder.when_last_row().assert_zero(
            (opcode - AB::Expr::from_canonical_usize(TERMINATE as usize))
                * (opcode - AB::Expr::from_canonical_usize(NOP as usize)),
        );

        // check accesses enabled
        builder.assert_eq(read1.enabled, read1_enabled_check);
        builder.assert_eq(read2.enabled, read2_enabled_check);
        builder.assert_eq(read3.enabled, read3_enabled_check);
        builder.assert_eq(write.enabled, write_enabled_check);

        // Turn on all interactions
        self.eval_interactions(
            builder,
            io,
            next_io,
            &operation_flags,
            AB::Expr::one() - is_cpu_opcode,
        );
    }
}
