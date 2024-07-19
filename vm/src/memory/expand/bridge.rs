use p3_air::{PairCol, VirtualPairCol};
use p3_field::Field;

use afs_stark_backend::interaction::{AirBridge, Interaction};

use crate::memory::expand::air::ExpandAir;
use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::{EXPAND_BUS, POSEIDON2_DIRECT_REQUEST_BUS};

fn interaction<const CHUNK: usize, F: Field>(
    sends: VirtualPairCol<F>,
    is_final: VirtualPairCol<F>,
    height: VirtualPairCol<F>,
    label: VirtualPairCol<F>,
    address_space: usize,
    hash: [usize; CHUNK],
) -> Interaction<F> {
    let mut fields = vec![
        is_final,
        VirtualPairCol::single_main(address_space),
        height,
        label,
    ];
    fields.extend(hash.map(VirtualPairCol::single_main));
    Interaction {
        fields,
        count: sends,
        argument_index: EXPAND_BUS,
    }
}
impl<const CHUNK: usize, F: Field> AirBridge<F> for ExpandAir<CHUNK> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let all_cols = (0..ExpandCols::<CHUNK, F>::get_width()).collect::<Vec<usize>>();
        let cols_numbered = ExpandCols::<CHUNK, usize>::from_slice(&all_cols);

        let child_height =
            VirtualPairCol::new_main(vec![(cols_numbered.parent_height, F::one())], F::neg_one());

        let mut poseidon2_fields = vec![];
        poseidon2_fields.extend(
            cols_numbered
                .left_child_hash
                .map(VirtualPairCol::single_main),
        );
        poseidon2_fields.extend(
            cols_numbered
                .right_child_hash
                .map(VirtualPairCol::single_main),
        );
        poseidon2_fields.extend(cols_numbered.parent_hash.map(VirtualPairCol::single_main));

        vec![
            interaction(
                VirtualPairCol::new_main(vec![(cols_numbered.direction, F::neg_one())], F::zero()),
                VirtualPairCol::new_main(
                    vec![(cols_numbered.direction, F::two().inverse().neg())],
                    F::two().inverse(),
                ),
                VirtualPairCol::single_main(cols_numbered.parent_height),
                VirtualPairCol::single_main(cols_numbered.parent_label),
                cols_numbered.address_space,
                cols_numbered.parent_hash,
            ),
            interaction(
                VirtualPairCol::single_main(cols_numbered.direction),
                VirtualPairCol::new_main(
                    vec![
                        (cols_numbered.direction, F::two().inverse().neg()),
                        (cols_numbered.left_is_final, F::one()),
                    ],
                    F::two().inverse(),
                ),
                child_height.clone(),
                VirtualPairCol::new(
                    vec![(PairCol::Main(cols_numbered.parent_label), F::two())],
                    F::zero(),
                ),
                cols_numbered.address_space,
                cols_numbered.left_child_hash,
            ),
            interaction(
                VirtualPairCol::single_main(cols_numbered.direction),
                VirtualPairCol::new_main(
                    vec![
                        (cols_numbered.direction, F::two().inverse().neg()),
                        (cols_numbered.right_is_final, F::one()),
                    ],
                    F::two().inverse(),
                ),
                child_height,
                VirtualPairCol::new(
                    vec![(PairCol::Main(cols_numbered.parent_label), F::two())],
                    F::one(),
                ),
                cols_numbered.address_space,
                cols_numbered.right_child_hash,
            ),
            Interaction {
                fields: poseidon2_fields,
                count: VirtualPairCol::constant(F::one()),
                argument_index: POSEIDON2_DIRECT_REQUEST_BUS,
            },
        ]
    }
}
