//! Hints emulation for the non-zkVM environment.

use alloc::vec::Vec;

#[cfg(feature = "std")]
pub use input::*;

#[cfg(feature = "std")]
mod input {
    use alloc::vec::Vec;
    use std::cell::RefCell;

    /// Simulated input stream on host
    pub enum HostInputStream {
        /// Read directly from stdin
        Stdin,
        /// Directly set from a test using [`set_hints`].
        Internal(Vec<Vec<u8>>),
    }

    impl HostInputStream {
        pub const fn new() -> Self {
            Self::Stdin
        }
    }

    impl Default for HostInputStream {
        fn default() -> Self {
            Self::new()
        }
    }

    thread_local! {
        /// Hint streams in the non-zkVM environment.
        pub static HINTS: RefCell<HostInputStream> = const { RefCell::new(HostInputStream::new()) };
        /// Current hint stream in the non-zkVM environment.
        pub static HINT_STREAM: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    /// Set the hints and reset the current hint stream.
    pub fn set_hints(hints: Vec<Vec<u8>>) {
        HINTS.replace(HostInputStream::Internal(
            hints
                .into_iter()
                .rev()
                .map(|v| {
                    let len = v.len() as u32;
                    len.to_le_bytes()
                        .into_iter()
                        .chain(v.iter().cloned())
                        .collect()
                })
                .collect(),
        ));
        HINT_STREAM.replace(Vec::new());
    }
}

/// Read the next hint stream from the hints.
pub fn hint_input() {
    #[cfg(feature = "std")]
    {
        HINTS.with_borrow_mut(|hints| match hints {
            HostInputStream::Stdin => {
                use std::io::Read;
                let mut buf = Vec::new();
                std::io::stdin()
                    .read_to_end(&mut buf)
                    .expect("Failed to read from stdin");
                let hint = [&(buf.len() as u32).to_le_bytes(), &buf[..]].concat();
                HINT_STREAM.replace(hint);
            }
            HostInputStream::Internal(hints) => {
                let hint = hints.pop().expect("No hint stream available");
                HINT_STREAM.replace(hint);
            }
        });
    }
    #[cfg(not(feature = "std"))]
    unimplemented!("hint_input not supported on no_std host")
}

/// Read the next `n` bytes from the hint stream.
pub fn read_n_bytes(_n: usize) -> Vec<u8> {
    #[cfg(feature = "std")]
    {
        HINT_STREAM.with_borrow_mut(|stream| stream.drain(.._n).collect())
    }
    #[cfg(not(feature = "std"))]
    {
        unimplemented!("hint_stream not supported on no_std host")
    }
}

/// Read the next 4 bytes from the hint stream as a `u32`.
pub fn read_u32() -> u32 {
    let bytes: Vec<u8> = read_n_bytes(4);
    u32::from_le_bytes(bytes.try_into().unwrap())
}

#[cfg(all(feature = "std", test, not(target_os = "zkvm")))]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::io::read_vec;

    #[test]
    fn test_read_hints() {
        set_hints(vec![vec![1, 2, 3, 4]; 3]);
        hint_input();
        assert_eq!(read_u32(), 4);
        hint_input();
        assert_eq!(read_n_bytes(8), vec![4, 0, 0, 0, 1, 2, 3, 4]);
        assert_eq!(read_vec(), vec![1, 2, 3, 4]);
    }
}
