use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::{PairCol, VirtualPairCol};
use p3_field::PrimeField64;

use super::{
    columns::CpuCols, CpuAir, ARITHMETIC_BUS, CPU_MAX_READS_PER_CYCLE,
    FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_BUS, FIELD_EXTENSION_INSTRUCTIONS, MEMORY_BUS,
    READ_INSTRUCTION_BUS,
};

impl<const WORD_SIZE: usize, F: PrimeField64> AirBridge<F> for CpuAir<WORD_SIZE> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let all_cols =
            (0..CpuCols::<WORD_SIZE, F>::get_width(self.options)).collect::<Vec<usize>>();
        let cols_numbered = CpuCols::<WORD_SIZE, usize>::from_slice(&all_cols, self.options);

        let mut interactions = vec![];

        // Interaction with program (bus 0)
        interactions.push(Interaction {
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
        });

        for (i, access_cols) in cols_numbered.aux.accesses.iter().enumerate() {
            let memory_cycle = VirtualPairCol::new(
                vec![(PairCol::Main(cols_numbered.io.timestamp), F::one())],
                F::from_canonical_usize(i),
            );
            let is_write = i >= CPU_MAX_READS_PER_CYCLE;

            let mut fields = vec![
                memory_cycle,
                VirtualPairCol::constant(F::from_bool(is_write)),
                VirtualPairCol::single_main(access_cols.address_space),
                VirtualPairCol::single_main(access_cols.address),
            ];
            for data_cell in access_cols.data.iter() {
                fields.push(VirtualPairCol::single_main(*data_cell));
            }
            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::diff_main(access_cols.enabled, access_cols.is_immediate),
                argument_index: MEMORY_BUS,
            });
        }

        // Interaction with arithmetic (bus 2)
        if self.options.field_arithmetic_enabled {
            let mut fields = vec![];

            fields.push(VirtualPairCol::single_main(cols_numbered.io.opcode));

            let accesses = cols_numbered.aux.accesses;
            fields.push(VirtualPairCol::single_main(accesses[0].data[0]));
            fields.push(VirtualPairCol::single_main(accesses[1].data[0]));
            fields.push(VirtualPairCol::single_main(
                accesses[CPU_MAX_READS_PER_CYCLE].data[0],
            ));

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::sum_main(
                    FIELD_ARITHMETIC_INSTRUCTIONS
                        .iter()
                        .map(|opcode| cols_numbered.aux.operation_flags[opcode])
                        .collect(),
                ),
                argument_index: ARITHMETIC_BUS,
            });
        }

        // Interaction with field extension arithmetic (bus 3)
        if self.options.field_extension_enabled {
            let fields = vec![
                VirtualPairCol::single_main(cols_numbered.io.opcode),
                VirtualPairCol::single_main(cols_numbered.io.op_a),
                VirtualPairCol::single_main(cols_numbered.io.op_b),
                VirtualPairCol::single_main(cols_numbered.io.op_c),
                VirtualPairCol::single_main(cols_numbered.io.d),
                VirtualPairCol::single_main(cols_numbered.io.e),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::sum_main(
                    FIELD_EXTENSION_INSTRUCTIONS
                        .iter()
                        .map(|opcode| cols_numbered.aux.operation_flags[opcode])
                        .collect(),
                ),
                argument_index: FIELD_EXTENSION_BUS,
            });
        }

        interactions
    }
}
