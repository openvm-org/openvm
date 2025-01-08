use openvm_stark_backend::p3_field::FieldAlgebra;

use super::{Array, Builder, Config, DslIr, Ext, Felt, MemIndex, Ptr, Usize, Var};

pub const DIGEST_SIZE: usize = 8;
pub const HASH_RATE: usize = 8;
pub const PERMUTATION_WIDTH: usize = 16;

impl<C: Config> Builder<C> {
    /// Applies the Poseidon2 permutation to the given array.
    ///
    /// [Reference](https://docs.rs/p3-poseidon2/latest/p3_poseidon2/struct.Poseidon2.html)
    pub fn poseidon2_permute(&mut self, array: &Array<C, Felt<C::F>>) -> Array<C, Felt<C::F>> {
        let output = match array {
            Array::Fixed(values) => {
                assert_eq!(values.borrow().len(), PERMUTATION_WIDTH);
                self.dyn_array::<Felt<C::F>>(Usize::from(PERMUTATION_WIDTH))
            }
            Array::Dyn(_, len) => self.dyn_array::<Felt<C::F>>(len.clone()),
        };
        self.operations.push(DslIr::Poseidon2PermuteBabyBear(
            output.clone(),
            array.clone(),
        ));
        output
    }

    /// Applies the Poseidon2 permutation to the given array.
    ///
    /// [Reference](https://docs.rs/p3-poseidon2/latest/p3_poseidon2/struct.Poseidon2.html)
    pub fn poseidon2_permute_mut(&mut self, array: &Array<C, Felt<C::F>>) {
        if let Array::Fixed(_) = array {
            panic!("Poseidon2 permutation is not allowed on fixed arrays");
        }
        self.operations.push(DslIr::Poseidon2PermuteBabyBear(
            array.clone(),
            array.clone(),
        ));
    }

    /// Applies the Poseidon2 compression function to the given array.
    ///
    /// [Reference](https://docs.rs/p3-symmetric/latest/p3_symmetric/struct.TruncatedPermutation.html)
    pub fn poseidon2_compress(
        &mut self,
        left: &Array<C, Felt<C::F>>,
        right: &Array<C, Felt<C::F>>,
    ) -> Array<C, Felt<C::F>> {
        let perm_width = PERMUTATION_WIDTH;
        let input = self.dyn_array(perm_width);
        for i in 0..DIGEST_SIZE {
            let a = self.get(left, i);
            let b = self.get(right, i);
            self.set(&input, i, a);
            self.set(&input, i + DIGEST_SIZE, b);
        }
        self.poseidon2_permute_mut(&input);
        input
    }

    /// Applies the Poseidon2 compression to the given array.
    ///
    /// [Reference](https://docs.rs/p3-symmetric/latest/p3_symmetric/struct.TruncatedPermutation.html)
    pub fn poseidon2_compress_x(
        &mut self,
        result: &Array<C, Felt<C::F>>,
        left: &Array<C, Felt<C::F>>,
        right: &Array<C, Felt<C::F>>,
    ) {
        self.operations.push(DslIr::Poseidon2CompressBabyBear(
            result.clone(),
            left.clone(),
            right.clone(),
        ));
    }

    /// Applies the Poseidon2 permutation to the given array.
    ///
    /// [Reference](https://docs.rs/p3-symmetric/latest/p3_symmetric/struct.PaddingFreeSponge.html)
    pub fn poseidon2_hash(&mut self, array: &Array<C, Felt<C::F>>) -> Array<C, Felt<C::F>> {
        let perm_width = PERMUTATION_WIDTH;
        let state: Array<C, Felt<C::F>> = self.dyn_array(perm_width);
        self.range(0, perm_width).for_each(|i, builder| {
            builder.set(&state, i, C::F::ZERO);
        });

        let break_flag: Var<_> = self.eval(C::N::ZERO);
        let last_index: Usize<_> = self.eval(array.len() - C::N::ONE);
        let hash_rate: Var<_> = self.eval(C::N::from_canonical_usize(HASH_RATE));

        self.range(0, array.len())
            .may_break()
            .step_by(HASH_RATE)
            .for_each(|i, builder| {
                builder
                    .if_eq(break_flag, C::N::ONE)
                    .then_may_break(|builder| builder.break_loop())?;
                // Insert elements of the chunk.
                builder
                    .range(0, hash_rate)
                    .may_break()
                    .for_each(|j, builder| {
                        let index = builder.eval_expr(i + j);
                        let element = builder.get(array, index);
                        builder.set_value(&state, j, element);
                        builder
                            .if_eq(index, last_index.clone())
                            .then_may_break(|builder| {
                                builder.assign(&break_flag, C::N::ONE);
                                builder.break_loop()
                            })
                    });

                builder.poseidon2_permute_mut(&state);
                Ok(())
            });

        state.truncate(self, Usize::from(DIGEST_SIZE));
        state
    }

    pub fn poseidon2_hash_x(
        &mut self,
        array: &Array<C, Array<C, Felt<C::F>>>,
    ) -> Array<C, Felt<C::F>> {
        self.cycle_tracker_start("poseidon2-hash");
        let perm_width = PERMUTATION_WIDTH;
        let state: Array<C, Felt<C::F>> = self.dyn_array(perm_width);
        self.range(0, perm_width).for_each(|i, builder| {
            builder.set(&state, i, C::F::ZERO);
        });

        let address = self.get_ref(&state, 0).ptr.address;
        let start: Var<_> = self.eval(address);
        let end: Var<_> = self.eval(address + C::N::from_canonical_usize(HASH_RATE));
        self.iter(array).for_each(|subarray, builder| {
            builder.iter(&subarray).for_each(|element, builder| {
                builder.cycle_tracker_start("poseidon2-hash-setup");
                builder.store(
                    Ptr { address },
                    MemIndex {
                        index: 0.into(),
                        offset: 0,
                        size: 1,
                    },
                    element,
                );
                builder.assign(&address, address + C::N::ONE);
                builder.cycle_tracker_end("poseidon2-hash-setup");
                builder.if_eq(address, end).then(|builder| {
                    builder.poseidon2_permute_mut(&state);
                    builder.assign(&address, start);
                });
            });
        });

        self.if_ne(address, start).then(|builder| {
            builder.poseidon2_permute_mut(&state);
        });

        state.truncate(self, Usize::from(DIGEST_SIZE));
        self.cycle_tracker_end("poseidon2-hash");
        state
    }

    pub fn poseidon2_hash_ext(
        &mut self,
        array: &Array<C, Array<C, Ext<C::F, C::EF>>>,
    ) -> Array<C, Felt<C::F>> {
        self.cycle_tracker_start("poseidon2-hash-ext");
        let hash_rate = HASH_RATE;
        let perm_width = PERMUTATION_WIDTH;
        let state: Array<C, Felt<C::F>> = self.dyn_array(perm_width);
        self.range(hash_rate, perm_width).for_each(|i, builder| {
            builder.set(&state, i, C::F::ZERO);
        });

        let idx: Var<_> = self.eval(C::N::ZERO);
        self.range(0, array.len()).for_each(|i, builder| {
            let subarray = builder.get(array, i);
            builder.range(0, subarray.len()).for_each(|j, builder| {
                let element = builder.get(&subarray, j);
                let felts = builder.ext2felt(element);
                for i in 0..4 {
                    let felt = builder.get(&felts, i);
                    builder.set_value(&state, idx, felt);
                    builder.assign(&idx, idx + C::N::ONE);
                    builder
                        .if_eq(idx, C::N::from_canonical_usize(HASH_RATE))
                        .then(|builder| {
                            builder.poseidon2_permute_mut(&state);
                            builder.assign(&idx, C::N::ZERO);
                        });
                }
            });
        });

        self.if_ne(idx, C::N::ZERO).then(|builder| {
            builder.poseidon2_permute_mut(&state);
        });

        state.truncate(self, Usize::from(DIGEST_SIZE));
        self.cycle_tracker_end("poseidon2-hash-ext");
        state
    }
}
