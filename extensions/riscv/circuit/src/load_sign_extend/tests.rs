use super::test_utils::*;

#[test]
fn negative_split_signed_load_tests() {
    assert_pranked_byte_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_halfword_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_word_fails(|core| core.data_most_sig_bit += F::ONE);
}
