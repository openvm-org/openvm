pub(crate) mod phantom {
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;
    use p3_field::{Field, PrimeField32};

    use crate::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };

    pub struct NativeHintInputSubEx;
    pub struct NativePrintSubEx;
    pub struct NativeHintBitsSubEx;

    impl<F: Field> PhantomSubExecutor<F> for NativeHintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams
                .hint_stream
                .push_back(F::from_canonical_usize(hint.len()));
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativePrintSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            _: F,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let addr_space = F::from_canonical_u16(c_upper);
            let value = memory.unsafe_read_cell(addr_space, a);
            println!("{}", value);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativeHintBitsSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let addr_space = F::from_canonical_u16(c_upper);
            let val = memory.unsafe_read_cell(addr_space, a);
            let mut val = val.as_canonical_u32();

            let len = b.as_canonical_u32();
            streams.hint_stream.clear();
            for _ in 0..len {
                streams
                    .hint_stream
                    .push_back(F::from_canonical_u32(val & 1));
                val >>= 1;
            }
            Ok(())
        }
    }
}
