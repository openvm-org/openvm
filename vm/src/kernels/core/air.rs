use std::borrow::Borrow;

use afs_primitives::{
    is_equal::{columns::IsEqualIoCols, IsEqualAir},
    sub_chip::SubAir,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use itertools::izip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{CoreAuxCols, CoreCols, CoreIoCols},
    CoreOptions, INST_WIDTH,
};
use crate::{
    arch::{instructions::CoreOpcode::*, ExecutionBridge},
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};

/// Air for the Core. Carries no state and does not own execution.
#[derive(Clone, Debug)]
pub struct CoreAir {
    pub options: CoreOptions,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,

    pub(super) offset: usize,
}

impl<F: Field> PartitionedBaseAir<F> for CoreAir {}
impl<F: Field> BaseAir<F> for CoreAir {
    fn width(&self) -> usize {
        CoreCols::<F>::get_width(self)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for CoreAir {
    fn num_public_values(&self) -> usize {
        self.options.num_public_values
    }
}

impl<AB: AirBuilderWithPublicValues + InteractionBuilder> Air<AB> for CoreAir {
    // TODO: continuation verification checks program counters match up [INT-1732]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        // TODO: move these public values to the connector chip?

        let inst_width = AB::F::from_canonical_u32(INST_WIDTH);

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CoreCols::from_slice(local, self);

        let CoreCols { io, aux } = local_cols;

        let CoreIoCols {
            timestamp,
            pc,
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        } = io;

        let CoreAuxCols {
            operation_flags,
            public_value_flags,
            reads,
            writes,
            read0_equals_read1,
            is_equal_aux,
            reads_aux_cols,
            writes_aux_cols,
            next_pc,
        } = aux;

        let [read1, read2, read3] = &reads;
        let [write] = &writes;

        // set correct operation flag
        for &flag in operation_flags.values() {
            builder.assert_bool(flag);
        }

        let mut is_core_opcode = AB::Expr::zero();
        let mut match_opcode = AB::Expr::zero();
        for (&opcode, &flag) in operation_flags.iter() {
            is_core_opcode += flag.into();
            match_opcode += flag * AB::F::from_canonical_usize(opcode as usize);
        }
        builder.assert_bool(is_core_opcode.clone());
        builder
            .when(is_core_opcode.clone())
            .assert_eq(opcode, match_opcode);

        // keep track of when memory accesses should be enabled
        let mut read1_enabled = AB::Expr::zero();
        let mut read2_enabled = AB::Expr::zero();
        let mut read3_enabled = AB::Expr::zero();
        let mut write_enabled = AB::Expr::zero();

        // LOADW: d[a] <- e[d[c] + b + d[f] * g]
        let loadw_flag = operation_flags[&LOADW];
        read1_enabled += loadw_flag.into();
        read2_enabled += loadw_flag.into();
        write_enabled += loadw_flag.into();

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
        read1_enabled += storew_flag.into();
        read2_enabled += storew_flag.into();
        write_enabled += storew_flag.into();

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
        read1_enabled += loadw2_flag.into();
        read2_enabled += loadw2_flag.into();
        read3_enabled += loadw2_flag.into();
        write_enabled += loadw2_flag.into();

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
        read1_enabled += storew2_flag.into();
        read2_enabled += storew2_flag.into();
        read3_enabled += storew2_flag.into();
        write_enabled += storew2_flag.into();

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
        read1_enabled += shintw_flag.into();
        write_enabled += shintw_flag.into();

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
        write_enabled += jal_flag.into();

        let mut when_jal = builder.when(jal_flag);

        when_jal.assert_eq(write.address_space, d);
        when_jal.assert_eq(write.pointer, a);
        when_jal.assert_eq(write.value, pc + inst_width);

        when_jal.when_transition().assert_eq(next_pc, pc + b);

        // BEQ: If d[a] = e[b], pc <- pc + c
        let beq_flag = operation_flags[&BEQ];
        read1_enabled += beq_flag.into();
        read2_enabled += beq_flag.into();

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
        read1_enabled += bne_flag.into();
        read2_enabled += bne_flag.into();

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

        // TERMINATE
        let terminate_flag = operation_flags[&TERMINATE];
        let mut when_terminate = builder.when(terminate_flag);
        when_terminate.when_transition().assert_eq(next_pc, pc);

        // PUBLISH

        let publish_flag = operation_flags[&PUBLISH];
        read1_enabled += publish_flag.into();
        read2_enabled += publish_flag.into();

        let mut sum_flags = AB::Expr::zero();
        let mut match_public_value_index = AB::Expr::zero();
        let mut match_public_value = AB::Expr::zero();
        for (i, &flag) in public_value_flags.iter().enumerate() {
            builder.assert_bool(flag);
            sum_flags = sum_flags + flag;
            match_public_value_index += flag * AB::F::from_canonical_usize(i);
            match_public_value += flag * builder.public_values()[i].into();
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

        let is_equal_io_cols = IsEqualIoCols {
            x: read1.value,
            y: read2.value,
            is_equal: read0_equals_read1,
        };
        SubAir::eval(&IsEqualAir, builder, is_equal_io_cols, is_equal_aux);

        // make sure program terminates or shards with NOP
        builder.when_last_row().assert_zero(
            (opcode - AB::Expr::from_canonical_usize(TERMINATE as usize))
                * (opcode - AB::Expr::from_canonical_usize(NOP as usize)),
        );

        // Turn on all interactions
        self.eval_interactions(builder, io, next_pc, &operation_flags);
    }
}
