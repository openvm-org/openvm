use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::{PairCol, VirtualPairCol};
use p3_field::PrimeField64;

use crate::memory::OpType;

use super::{
    columns::CpuCols, CpuAir, ARITHMETIC_BUS, FIELD_ARITHMETIC_INSTRUCTIONS, MAX_READS_PER_CYCLE,
    MAX_WRITES_PER_CYCLE, MEMORY_BUS, READ_INSTRUCTION_BUS,
};

fn access_cycle<F: PrimeField64>(
    clock_cycle: usize,
    op_type: OpType,
    index: usize,
) -> VirtualPairCol<F> {
    VirtualPairCol::new(
        vec![(
            PairCol::Main(clock_cycle),
            F::from_canonical_usize(MAX_READS_PER_CYCLE + MAX_WRITES_PER_CYCLE),
        )],
        F::from_canonical_usize(
            index
                + if op_type == OpType::Write {
                    MAX_READS_PER_CYCLE
                } else {
                    0
                },
        ),
    )
}

impl<F: PrimeField64> AirBridge<F> for CpuAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let all_cols = (0..CpuCols::<F>::get_width(self.options)).collect::<Vec<usize>>();
        let cols_numbered = CpuCols::<usize>::from_slice(&all_cols, self.options);

        let interactions = vec![
            // Interaction with program (bus 0)
            Interaction {
                fields: vec![
                    VirtualPairCol::single_main(cols_numbered.io.pc),
                    VirtualPairCol::single_main(cols_numbered.io.opcode),
                    VirtualPairCol::single_main(cols_numbered.io.op_a),
                    VirtualPairCol::single_main(cols_numbered.io.op_b),
                    VirtualPairCol::single_main(cols_numbered.io.op_c),
                    VirtualPairCol::single_main(cols_numbered.io.d),
                    VirtualPairCol::single_main(cols_numbered.io.e),
                ],
                count: VirtualPairCol::constant(F::one()),
                argument_index: READ_INSTRUCTION_BUS,
            },
            // Interactions with memory (bus 1)
            Interaction {
                fields: vec![
                    access_cycle(cols_numbered.io.clock_cycle, OpType::Read, 0),
                    VirtualPairCol::constant(F::zero()),
                    VirtualPairCol::single_main(cols_numbered.aux.read1.address_space),
                    VirtualPairCol::single_main(cols_numbered.aux.read1.address),
                    VirtualPairCol::single_main(cols_numbered.aux.read1.data),
                ],
                count: VirtualPairCol::diff_main(
                    cols_numbered.aux.read1.enabled,
                    cols_numbered.aux.read1.is_immediate,
                ),
                argument_index: MEMORY_BUS,
            },
            Interaction {
                fields: vec![
                    access_cycle(cols_numbered.io.clock_cycle, OpType::Read, 1),
                    VirtualPairCol::constant(F::zero()),
                    VirtualPairCol::single_main(cols_numbered.aux.read2.address_space),
                    VirtualPairCol::single_main(cols_numbered.aux.read2.address),
                    VirtualPairCol::single_main(cols_numbered.aux.read2.data),
                ],
                count: VirtualPairCol::diff_main(
                    cols_numbered.aux.read2.enabled,
                    cols_numbered.aux.read2.is_immediate,
                ),
                argument_index: MEMORY_BUS,
            },
            Interaction {
                fields: vec![
                    access_cycle(cols_numbered.io.clock_cycle, OpType::Write, 0),
                    VirtualPairCol::constant(F::one()),
                    VirtualPairCol::single_main(cols_numbered.aux.write.address_space),
                    VirtualPairCol::single_main(cols_numbered.aux.write.address),
                    VirtualPairCol::single_main(cols_numbered.aux.write.data),
                ],
                count: VirtualPairCol::diff_main(
                    cols_numbered.aux.write.enabled,
                    cols_numbered.aux.write.is_immediate,
                ),
                argument_index: MEMORY_BUS,
            },
            // Interaction with arithmetic (bus 2)
            if self.options.field_arithmetic_enabled {
                Interaction {
                    fields: vec![
                        VirtualPairCol::single_main(cols_numbered.io.opcode),
                        VirtualPairCol::single_main(cols_numbered.aux.read1.data),
                        VirtualPairCol::single_main(cols_numbered.aux.read2.data),
                        VirtualPairCol::single_main(cols_numbered.aux.write.data),
                    ],
                    count: VirtualPairCol::sum_main(
                        FIELD_ARITHMETIC_INSTRUCTIONS
                            .iter()
                            .map(|opcode| cols_numbered.aux.operation_flags[opcode])
                            .collect(),
                    ),
                    argument_index: ARITHMETIC_BUS,
                }
            } else {
                Interaction {
                    fields: vec![],
                    count: VirtualPairCol::constant(F::zero()),
                    argument_index: ARITHMETIC_BUS,
                }
            },
        ];

        interactions
    }
}
