use core::mem::MaybeUninit;

use openvm_platform::WORD_SIZE;
#[cfg(target_os = "zkvm")]
use openvm_rv32im_guest::hint_buffer_u32;

use super::hint_store_word;
use crate::serde::WordRead;

/// Provides a Reader for reading serialized data from the hint stream.
#[derive(Copy, Clone)]
pub struct Reader {
    /// The number of bytes remaining in the hint stream.
    pub bytes_remaining: usize,
}

impl Reader {
    /// When a new [Reader] is constructed, the hint stream
    /// is reset to the next vector of bytes from the input
    /// stream. The program will fail if there is no next
    /// stream in the input stream.
    pub fn new() -> Self {
        super::hint_input();
        let bytes_remaining = super::read_u32() as usize;
        Self { bytes_remaining }
    }
}

impl WordRead for Reader {
    fn read_words(&mut self, words: &mut [u32]) -> crate::serde::Result<()> {
        let num_words = words.len();
        if let Some(new_remaining) = self.bytes_remaining.checked_sub(num_words * WORD_SIZE) {
            #[cfg(target_os = "zkvm")]
            hint_buffer_u32!(words.as_mut_ptr(), words.len());
            #[cfg(not(target_os = "zkvm"))]
            {
                for w in words.iter_mut() {
                    hint_store_word(w as *mut u32);
                }
            }
            self.bytes_remaining = new_remaining;
            Ok(())
        } else {
            Err(crate::serde::Error::DeserializeUnexpectedEnd)
        }
    }

    fn read_padded_bytes(&mut self, bytes: &mut [u8]) -> crate::serde::Result<()> {
        if self.bytes_remaining < bytes.len() {
            return Err(crate::serde::Error::DeserializeUnexpectedEnd);
        }
        let mut num_padded_bytes = bytes.len();
        #[cfg(target_os = "zkvm")]
        hint_buffer_u32!(bytes as *mut [u8] as *mut u32, num_padded_bytes / WORD_SIZE);
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut words = bytes.chunks_exact_mut(WORD_SIZE);
            for word in &mut words {
                hint_store_word(word as *mut [u8] as *mut u32);
            }
        }

        let remainder = bytes.chunks_exact_mut(WORD_SIZE).into_remainder();
        if !remainder.is_empty() {
            num_padded_bytes += WORD_SIZE - remainder.len();
            let mut padded = MaybeUninit::<[u8; WORD_SIZE]>::uninit();
            hint_store_word(padded.as_mut_ptr() as *mut u32);
            let padded = unsafe { padded.assume_init() };
            remainder.copy_from_slice(&padded[..remainder.len()]);
        }
        // If we reached EOF, then we set to 0.
        // Otherwise, we need to subtract the padding as well.
        self.bytes_remaining = self.bytes_remaining.saturating_sub(num_padded_bytes);
        Ok(())
    }
}
