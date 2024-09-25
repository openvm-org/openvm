use std::borrow::Borrow;

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecIoCols, IsEqualVecAir},
    sub_chip::SubAir,
};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{CoreAuxCols, CoreCols, CoreIoCols},
    CoreOptions, INST_WIDTH, WORD_SIZE,
};
use crate::{
    arch::{
        bridge::ExecutionBridge,
        instructions::{Opcode::*, OpcodeEncoder, CORE_INSTRUCTIONS},
    },
    memory::{offline_checker::MemoryBridge, MemoryAddress},
};

/// Air for the Core. Carries no state and does not own execution.
#[derive(Clone, Debug)]
pub struct CoreAir {
    pub options: CoreOptions,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub opcode_encoder: OpcodeEncoder<4, 6>,
}

impl<F: Field> BaseAir<F> for CoreAir {
    fn width(&self) -> usize {
        CoreCols::<F>::get_width(self)
    }
}

// TODO[osama]: here, there should be some relation enforced between the timestamp for the cpu and the memory timestamp
// TODO[osama]: also, rename to clk
impl<AB: AirBuilderWithPublicValues + InteractionBuilder> Air<AB> for CoreAir {
    // TODO: continuation verification checks program counters match up [INT-1732]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        // TODO: move these public values to the connector chip?

        let inst_width = AB::F::from_canonical_usize(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CoreCols::from_slice(local, self);

        let CoreCols { io, aux } = local_cols;

        let CoreIoCols {
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

        let CoreAuxCols {
            operation_flags,
            public_value_flags,
            reads,
            writes,
            read0_equals_read1,
            is_equal_vec_aux,
            reads_aux_cols,
            writes_aux_cols,
            next_pc,
        } = aux;

        let [read1, read2, read3] = &reads;
        let [write] = &writes;

        let encoder = self.opcode_encoder.initialize(builder, operation_flags);

        let mut match_opcode = AB::Expr::zero();
        for opcode in CORE_INSTRUCTIONS {
            let flag = encoder.expression_for(opcode);
            match_opcode += flag * AB::F::from_canonical_usize(opcode as usize);
        }
        builder.assert_eq(opcode, match_opcode);

        // keep track of when memory accesses should be enabled
        let mut read1_enabled = AB::Expr::zero();
        let mut read2_enabled = AB::Expr::zero();
        let mut read3_enabled = AB::Expr::zero();
        let mut write_enabled = AB::Expr::zero();

        // LOADW: d[a] <- e[d[c] + b + d[f] * g]
        let loadw_flag = encoder.expression_for(LOADW);
        read1_enabled += loadw_flag.clone();
        read2_enabled += loadw_flag.clone();
        write_enabled += loadw_flag.clone();

        let mut when_loadw = encoder.when(builder, LOADW);

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
        let storew_flag = encoder.expression_for(STOREW);
        read1_enabled += storew_flag.clone();
        read2_enabled += storew_flag.clone();
        write_enabled += storew_flag.clone();

        let mut when_storew = encoder.when(builder, STOREW);
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
        let loadw2_flag = encoder.expression_for(LOADW2);
        read1_enabled += loadw2_flag.clone();
        read2_enabled += loadw2_flag.clone();
        read3_enabled += loadw2_flag.clone();
        write_enabled += loadw2_flag.clone();

        let mut when_loadw2 = encoder.when(builder, LOADW2);

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
        let storew2_flag = encoder.expression_for(STOREW2);
        read1_enabled += storew2_flag.clone();
        read2_enabled += storew2_flag.clone();
        read3_enabled += storew2_flag.clone();
        write_enabled += storew2_flag.clone();

        let mut when_storew2 = encoder.when(builder, STOREW2);
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
        let shintw_flag = encoder.expression_for(SHINTW);
        read1_enabled += shintw_flag.clone();
        write_enabled += shintw_flag.clone();

        let mut when_shintw = encoder.when(builder, SHINTW);
        when_shintw.assert_eq(read1.address_space, d);
        when_shintw.assert_eq(read1.pointer, a);

        when_shintw.assert_eq(write.address_space, e);
        when_shintw.assert_eq(read1.value, write.pointer - b);

        when_shintw
            .when_transition()
            .assert_eq(next_pc, pc + inst_width);

        // JAL: d[a] <- pc + INST_WIDTH, pc <- pc + b
        let jal_flag = encoder.expression_for(JAL);
        write_enabled += jal_flag;

        let mut when_jal = encoder.when(builder, JAL);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.pointer, a);
        when_jal.assert_eq(write.value, pc + inst_width);

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let beq_flag = encoder.expression_for(BEQ);
        read1_enabled += beq_flag.clone();
        read2_enabled += beq_flag.clone();

        let mut when_beq = encoder.when(builder, BEQ);

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
        let bne_flag = encoder.expression_for(BNE);
        read1_enabled += bne_flag.clone();
        read2_enabled += bne_flag.clone();

        let mut when_bne = encoder.when(builder, BNE);

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
        let mut when_nop = encoder.when(builder, NOP);
        when_nop.when_transition().assert_eq(next_pc, pc);

        // TERMINATE
        let mut when_terminate = encoder.when(builder, TERMINATE);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // PUBLISH

        let publish_flag = encoder.expression_for(PUBLISH);
        read1_enabled += publish_flag.clone();
        read2_enabled += publish_flag.clone();

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

        let mut op_timestamp: AB::Expr = timestamp.into();

        let reads_enabled = [read1_enabled, read2_enabled, read3_enabled];
        for (read, read_aux_cols, enabled) in izip!(&reads, reads_aux_cols, reads_enabled) {
            self.memory_bridge
                .read_or_immediate(
                    MemoryAddress::new(read.address_space, read.pointer),
                    read.value,
                    op_timestamp.clone(),
                    &read_aux_cols,
                )
                .eval(builder, enabled.clone());
            op_timestamp += enabled.clone();
        }

        let writes_enabled = [write_enabled];
        for (write, write_aux_cols, enabled) in izip!(&writes, writes_aux_cols, writes_enabled) {
            self.memory_bridge
                .write(
                    MemoryAddress::new(write.address_space, write.pointer),
                    [write.value],
                    op_timestamp.clone(),
                    &write_aux_cols,
                )
                .eval(builder, enabled.clone());
            op_timestamp += enabled.clone();
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

        // make sure program terminates or shards with NOP
        builder.when_last_row().assert_zero(
            (opcode - AB::Expr::from_canonical_usize(TERMINATE as usize))
                * (opcode - AB::Expr::from_canonical_usize(NOP as usize)),
        );

        // Turn on all interactions
        self.eval_interactions(builder, io, next_pc, &encoder);
    }
}
