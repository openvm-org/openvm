use p3_air::{PairCol, VirtualPairCol};
use p3_field::Field;

use afs_stark_backend::interaction::{AirBridge, Interaction};

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::{ExpandAir, EXPAND_BUS};

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

        let compress_sign =
            VirtualPairCol::new_main(vec![(cols_numbered.is_compress, F::two())], F::neg_one());
        let expand_sign = VirtualPairCol::new_main(
            vec![(cols_numbered.is_compress, F::neg(F::two()))],
            F::one(),
        );
        let child_height =
            VirtualPairCol::new_main(vec![(cols_numbered.parent_height, F::one())], F::neg_one());

        vec![
            interaction(
                VirtualPairCol::new_main(
                    vec![(cols_numbered.multiplicity, F::neg_one())],
                    F::zero(),
                ),
                VirtualPairCol::single_main(cols_numbered.is_compress),
                VirtualPairCol::single_main(cols_numbered.parent_height),
                VirtualPairCol::single_main(cols_numbered.parent_label),
                cols_numbered.address_space,
                cols_numbered.parent_hash,
            ),
            interaction(
                VirtualPairCol::single_main(cols_numbered.multiplicity),
                VirtualPairCol::sum_main(vec![
                    cols_numbered.is_compress,
                    cols_numbered.left_is_final,
                ]),
                child_height.clone(),
                VirtualPairCol::new(
                    vec![(PairCol::Main(cols_numbered.is_compress), F::two())],
                    F::zero(),
                ),
                cols_numbered.address_space,
                cols_numbered.left_child_hash,
            ),
            interaction(
                VirtualPairCol::single_main(cols_numbered.multiplicity),
                VirtualPairCol::sum_main(vec![
                    cols_numbered.is_compress,
                    cols_numbered.right_is_final,
                ]),
                child_height,
                VirtualPairCol::new(
                    vec![(PairCol::Main(cols_numbered.is_compress), F::two())],
                    F::one(),
                ),
                cols_numbered.address_space,
                cols_numbered.right_child_hash,
            ),
        ]
    }
}
