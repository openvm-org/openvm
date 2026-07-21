use std::collections::TryReserveError;

use openvm_platform::WORD_SIZE;

/// The byte stream read by guest hint-store instructions.
///
/// ```text
/// Input record: [8-byte little-endian payload length | payload | zero padding]
/// Hint:         [hint bytes]
/// ```
///
/// The payload is padded with zeros to a multiple of 8 bytes.
#[derive(Clone, Default)]
pub struct HintStream {
    /// The current input payload or hint.
    bytes: Vec<u8>,
    /// Offset of the next unread byte in the stream exposed to the guest.
    position: usize,
    /// Whether `bytes` contains an input payload rather than a hint.
    is_input: bool,
}

impl HintStream {
    /// Sets an input payload, taking ownership of its bytes without copying them.
    #[inline]
    pub fn set_input(&mut self, bytes: Vec<u8>) {
        self.bytes = bytes;
        self.position = 0;
        self.is_input = true;
    }

    /// Sets a hint, taking ownership of its bytes without copying them.
    #[inline]
    pub fn set_hint(&mut self, bytes: Vec<u8>) {
        self.bytes = bytes;
        self.position = 0;
        self.is_input = false;
    }

    /// Copies hint bytes from a slice, reusing existing capacity when possible.
    #[inline]
    pub fn set_hint_from_slice(&mut self, bytes: &[u8]) {
        self.clear();
        self.bytes.extend_from_slice(bytes);
    }

    /// Replaces the hint with bytes from an iterator, reusing existing capacity when possible.
    #[inline]
    pub fn set_hint_from_iter(&mut self, bytes: impl IntoIterator<Item = u8>) {
        self.clear();
        self.bytes.extend(bytes);
    }

    /// Replaces the hint after reserving enough space for all of its bytes.
    #[inline]
    pub fn try_set_hint_from_iter(
        &mut self,
        len: usize,
        bytes: impl IntoIterator<Item = u8>,
    ) -> Result<(), TryReserveError> {
        self.clear();
        self.bytes.try_reserve(len)?;
        self.bytes.extend(bytes);
        Ok(())
    }

    /// Removes all bytes and resets the stream.
    #[inline]
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.position = 0;
        self.is_input = false;
    }

    /// Returns the number of unread bytes.
    ///
    /// For input records, this includes the payload-length prefix and padding.
    #[inline]
    pub fn remaining(&self) -> usize {
        let stream_len = if self.is_input {
            // Input reads begin with the payload length and end on a word boundary.
            WORD_SIZE + self.bytes.len().next_multiple_of(WORD_SIZE)
        } else {
            self.bytes.len()
        };
        stream_len.saturating_sub(self.position)
    }

    /// Copies exactly `dst.len()` bytes into `dst` and advances the stream.
    ///
    /// # Panics
    ///
    /// Panics if the stream contains fewer than `dst.len()` bytes.
    #[inline]
    pub fn copy_to_slice(&mut self, dst: &mut [u8]) {
        assert!(self.remaining() >= dst.len());

        let end_position = self.position + dst.len();
        if !self.is_input {
            dst.copy_from_slice(&self.bytes[self.position..end_position]);
            self.position = end_position;
            return;
        }

        let mut position = self.position;
        let mut dst = dst;

        // Copy any unread bytes from the payload-length prefix.
        if position < WORD_SIZE {
            let payload_length = (self.bytes.len() as u64).to_le_bytes();
            let count = (WORD_SIZE - position).min(dst.len());
            dst[..count].copy_from_slice(&payload_length[position..position + count]);
            position += count;
            dst = &mut dst[count..];
        }

        // Input positions include the length prefix before the payload.
        let input_payload_end = WORD_SIZE + self.bytes.len();
        if !dst.is_empty() && position < input_payload_end {
            let payload_offset = position - WORD_SIZE;
            let count = (input_payload_end - position).min(dst.len());
            dst[..count].copy_from_slice(&self.bytes[payload_offset..payload_offset + count]);
            dst = &mut dst[count..];
        }

        // After the prefix and payload, any requested bytes are zero padding.
        dst.fill(0);
        self.position = end_position;
    }
}

#[cfg(test)]
mod tests {
    use super::HintStream;

    fn expected_input_bytes(payload: &[u8]) -> Vec<u8> {
        let mut expected = Vec::with_capacity(8 + payload.len().next_multiple_of(8));
        expected.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        expected.extend_from_slice(payload);
        expected.resize(expected.len().next_multiple_of(8), 0);
        expected
    }

    #[test]
    fn input_is_not_copied_and_includes_length_and_padding() {
        for length in [0, 1, 7, 8, 9, 8_191, 8_192] {
            let payload = (0..length).map(|i| (i % 251) as u8).collect::<Vec<_>>();
            let payload_ptr = payload.as_ptr();
            let expected = expected_input_bytes(&payload);
            let mut stream = HintStream::default();

            stream.set_input(payload);

            assert_eq!(stream.bytes.as_ptr(), payload_ptr);
            let mut actual = Vec::with_capacity(expected.len());
            for chunk_size in [1, 3, 8, 13].into_iter().cycle() {
                if actual.len() == expected.len() {
                    break;
                }
                let count = chunk_size.min(expected.len() - actual.len());
                let mut chunk = vec![0xa5; count];
                stream.copy_to_slice(&mut chunk);
                actual.extend(chunk);
            }
            assert_eq!(actual, expected);
            assert_eq!(stream.remaining(), 0);
        }
    }

    #[test]
    fn setting_a_hint_discards_the_partially_read_input() {
        let mut stream = HintStream::default();
        stream.set_input(vec![1, 2, 3]);
        stream.copy_to_slice(&mut [0; 5]);

        stream.set_hint(vec![9, 8, 7]);

        let mut actual = [0; 3];
        stream.copy_to_slice(&mut actual);
        assert_eq!(actual, [9, 8, 7]);
        assert_eq!(stream.remaining(), 0);
    }

    #[test]
    fn cloned_streams_advance_independently() {
        let payload = (1u8..=9).collect::<Vec<_>>();
        let expected = expected_input_bytes(&payload);
        let mut original = HintStream::default();
        original.set_input(payload);
        original.copy_to_slice(&mut [0; 10]);

        let mut cloned = original.clone();
        let mut original_suffix = vec![0; original.remaining()];
        original.copy_to_slice(&mut original_suffix);
        let mut cloned_chunk = [0; 3];
        cloned.copy_to_slice(&mut cloned_chunk);

        assert_eq!(original_suffix, expected[10..]);
        assert_eq!(cloned_chunk, expected[10..13]);
    }
}
