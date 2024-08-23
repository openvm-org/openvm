use std::{array::from_fn, borrow::Borrow};

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
    timestamp_delta, CpuOptions, CPU_MAX_ACCESSES_PER_CYCLE, CPU_MAX_READS_PER_CYCLE, INST_WIDTH,
};
use crate::{
    arch::{bridge::ExecutionBus, instructions::Opcode::*},
    memory::{
        offline_checker::bridge::{MemoryBridge, MemoryOfflineChecker},
        MemoryAddress,
    },
};

/// Air for the CPU. Carries no state and does not own execution.
#[derive(Clone, Debug)]
pub struct CpuAir<const WORD_SIZE: usize> {
    pub options: CpuOptions,
    pub execution_bus: ExecutionBus,
    pub memory_offline_checker: MemoryOfflineChecker,
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for CpuAir<WORD_SIZE> {
    fn width(&self) -> usize {
        CpuCols::<WORD_SIZE, F>::get_width(self)
    }
}

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    fn assert_compose<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        word: [impl Into<AB::Expr>; WORD_SIZE],
        field_elem: AB::Expr,
    ) {
        let mut iter = word.into_iter();
        builder.assert_eq(iter.next().unwrap(), field_elem);
        for cell in iter.take(WORD_SIZE - 1) {
            builder.assert_zero(cell);
        }
    }
}

// TODO[osama]: here, there should be some relation enforced between the timestamp for the cpu and the memory timestamp
// TODO[osama]: also, rename to clk
impl<const WORD_SIZE: usize, AB: AirBuilderWithPublicValues + InteractionBuilder> Air<AB>
    for CpuAir<WORD_SIZE>
{
    // TODO: continuation verification checks program counters match up [INT-1732]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let start_pc = pis[0];
        let end_pc = pis[1];

        let inst_width = AB::F::from_canonical_usize(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CpuCols::<WORD_SIZE, AB::Var>::from_slice(local, self);

        let next = main.row_slice(1);
        let next: &[AB::Var] = (*next).borrow();
        let next_cols = CpuCols::<WORD_SIZE, AB::Var>::from_slice(next, self);
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
            mem_ops,
            read0_equals_read1,
            is_equal_vec_aux,
            mem_oc_aux_cols,
        } = aux;

        let read1 = &mem_ops[0];
        let read2 = &mem_ops[1];
        let read3 = &mem_ops[2];
        let write = &mem_ops[CPU_MAX_READS_PER_CYCLE];

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
            is_cpu_opcode = is_cpu_opcode + flag;
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

        when_loadw.assert_eq(read1.addr_space, d);
        when_loadw.assert_eq(read1.pointer, c);

        when_loadw.assert_eq(read2.addr_space, e);
        self.assert_compose(&mut when_loadw, read1.cell.data, read2.pointer - b);

        when_loadw.assert_eq(write.addr_space, d);
        when_loadw.assert_eq(write.pointer, a);

        for i in 0..WORD_SIZE {
            when_loadw.assert_eq(write.cell.data[i], read2.cell.data[i]);
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
        when_storew.assert_eq(read1.addr_space, d);
        when_storew.assert_eq(read1.pointer, c);

        when_storew.assert_eq(read2.addr_space, d);
        when_storew.assert_eq(read2.pointer, a);

        when_storew.assert_eq(write.addr_space, e);
        self.assert_compose(&mut when_storew, read1.cell.data, write.pointer - b);
        for i in 0..WORD_SIZE {
            when_storew.assert_eq(write.cell.data[i], read2.cell.data[i]);
        }

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

        when_loadw2.assert_eq(read1.addr_space, d);
        when_loadw2.assert_eq(read1.pointer, c);

        when_loadw2.assert_eq(read2.addr_space, d);
        when_loadw2.assert_eq(read2.pointer, f);

        when_loadw2.assert_eq(read3.addr_space, e);
        let addr_diff =
            from_fn::<AB::Expr, WORD_SIZE, _>(|i| read1.cell.data[i] + g * read2.cell.data[i]);
        self.assert_compose(&mut when_loadw2, addr_diff, read3.pointer - b);

        when_loadw2.assert_eq(write.addr_space, d);
        when_loadw2.assert_eq(write.pointer, a);

        for i in 0..WORD_SIZE {
            when_loadw2.assert_eq(write.cell.data[i], read3.cell.data[i]);
        }

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
        when_storew2.assert_eq(read1.addr_space, d);
        when_storew2.assert_eq(read1.pointer, c);

        when_storew2.assert_eq(read2.addr_space, d);
        when_storew2.assert_eq(read2.pointer, a);

        when_storew2.assert_eq(read3.addr_space, d);
        when_storew2.assert_eq(read3.pointer, f);

        when_storew2.assert_eq(write.addr_space, e);
        let addr_diff =
            from_fn::<AB::Expr, WORD_SIZE, _>(|i| read1.cell.data[i] + g * read3.cell.data[i]);
        self.assert_compose(&mut when_storew2, addr_diff, write.pointer - b);
        for i in 0..WORD_SIZE {
            when_storew2.assert_eq(write.cell.data[i], read2.cell.data[i]);
        }

        when_storew2
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // SHINTW: e[d[a] + b] <- ?
        let shintw_flag = operation_flags[&SHINTW];
        read1_enabled_check = read1_enabled_check + shintw_flag;
        write_enabled_check = write_enabled_check + shintw_flag;

        let mut when_shintw = builder.when(shintw_flag);
        when_shintw.assert_eq(read1.addr_space, d);
        when_shintw.assert_eq(read1.pointer, a);

        when_shintw.assert_eq(write.addr_space, e);
        self.assert_compose(&mut when_shintw, read1.cell.data, write.pointer - b);

        when_shintw
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let jal_flag = operation_flags[&JAL];
        write_enabled_check = write_enabled_check + jal_flag;

        let mut when_jal = builder.when(jal_flag);

        when_jal.assert_eq(write.addr_space, d);
        when_jal.assert_eq(write.pointer, a);
        self.assert_compose(&mut when_jal, write.cell.data, pc + inst_width);

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let beq_flag = operation_flags[&BEQ];
        read1_enabled_check = read1_enabled_check + beq_flag;
        read2_enabled_check = read2_enabled_check + beq_flag;

        let mut when_beq = builder.when(beq_flag);

        when_beq.assert_eq(read1.addr_space, d);
        when_beq.assert_eq(read1.pointer, a);

        when_beq.assert_eq(read2.addr_space, e);
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

        when_bne.assert_eq(read1.addr_space, d);
        when_bne.assert_eq(read1.pointer, a);

        when_bne.assert_eq(read2.addr_space, e);
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
        self.assert_compose(&mut when_publish, read1.cell.data, match_public_value_index);
        self.assert_compose(&mut when_publish, read2.cell.data, match_public_value);

        when_publish.assert_eq(read1.addr_space, d);
        when_publish.assert_eq(read1.pointer, a);

        when_publish.assert_eq(read2.addr_space, e);
        when_publish.assert_eq(read2.pointer, b);

        let mut op_timestamp: AB::Expr = io.timestamp.into();
        let mut memory_bridge = MemoryBridge::new(self.memory_offline_checker, mem_oc_aux_cols);
        for op in &mem_ops[0..CPU_MAX_READS_PER_CYCLE] {
            memory_bridge
                .read::<AB::Expr>(
                    MemoryAddress::new(op.addr_space, op.pointer),
                    op.cell.data,
                    op_timestamp.clone(),
                )
                .eval(builder, op.enabled);
            op_timestamp += op.enabled.into();
        }
        for op in &mem_ops[CPU_MAX_READS_PER_CYCLE..CPU_MAX_ACCESSES_PER_CYCLE] {
            memory_bridge
                .write(
                    MemoryAddress::new(op.addr_space, op.pointer),
                    op.cell.data,
                    op_timestamp.clone(),
                )
                .eval(builder, op.enabled);
            op_timestamp += op.enabled.into();
        }

        // evaluate equality between read1 and read2

        let is_equal_vec_io_cols = IsEqualVecIoCols {
            x: read1.cell.data.to_vec(),
            y: read2.cell.data.to_vec(),
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
