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

        let mut poseidon2_fields = vec![];
        poseidon2_fields.extend(
            cols_numbered
                .child_hashes
                .concat()
                .into_iter()
                .map(VirtualPairCol::single_main),
        );
        poseidon2_fields.extend(cols_numbered.parent_hash.map(VirtualPairCol::single_main));

        let mut interactions = vec![Interaction {
            fields: poseidon2_fields,
            count: VirtualPairCol::constant(F::one()),
            argument_index: POSEIDON2_DIRECT_REQUEST_BUS,
        }];

        interactions.push(interaction(
            VirtualPairCol::new_main(vec![(cols_numbered.direction, F::neg_one())], F::zero()),
            VirtualPairCol::new_main(
                vec![(cols_numbered.direction, F::neg(F::two().inverse()))],
                F::two().inverse(),
            ),
            VirtualPairCol::single_main(cols_numbered.parent_height),
            VirtualPairCol::single_main(cols_numbered.parent_label),
            cols_numbered.address_space,
            cols_numbered.parent_hash,
        ));

        let child_height =
            VirtualPairCol::new_main(vec![(cols_numbered.parent_height, F::one())], F::neg_one());
        for i in 0..2 {
            interactions.push(interaction(
                VirtualPairCol::single_main(cols_numbered.direction),
                VirtualPairCol::new_main(
                    vec![
                        (cols_numbered.direction, F::neg(F::two().inverse())),
                        (cols_numbered.are_final[i], F::one()),
                    ],
                    F::two().inverse(),
                ),
                child_height.clone(),
                // label = (2 * parent_label) + i
                VirtualPairCol::new(
                    vec![(PairCol::Main(cols_numbered.parent_label), F::two())],
                    F::from_canonical_usize(i),
                ),
                cols_numbered.address_space,
                cols_numbered.child_hashes[i],
            ));
        }
        interactions
    }
}
