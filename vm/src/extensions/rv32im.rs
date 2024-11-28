pub(crate) mod phantom {
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;
    use p3_field::{Field, PrimeField32};

    use crate::{
        arch::{PhantomSubExecutor, Streams},
        rv32im::adapters::unsafe_read_rv32_register,
        system::memory::MemoryController,
    };

    pub struct Rv32HintInputSubEx;
    pub struct Rv32PrintStrSubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv32HintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let mut hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                (hint.len() as u32)
                    .to_le_bytes()
                    .iter()
                    .map(|b| F::from_canonical_u8(*b)),
            );
            // Extend by 0 for 4 byte alignment
            let capacity = hint.len().div_ceil(4) * 4;
            hint.resize(capacity, F::ZERO);
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32PrintStrSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = unsafe_read_rv32_register(memory, a);
            let rs1 = unsafe_read_rv32_register(memory, b);
            let bytes = (0..rs1)
                .map(|i| -> eyre::Result<u8> {
                    let val = memory.unsafe_read_cell(F::TWO, F::from_canonical_u32(rd + i));
                    let byte: u8 = val.as_canonical_u32().try_into()?;
                    Ok(byte)
                })
                .collect::<eyre::Result<Vec<u8>>>()?;
            let peeked_str = String::from_utf8(bytes)?;
            println!("{peeked_str}");
            Ok(())
        }
    }
}
