use std::{fmt::Debug, hash::Hash};

use pest::iterators::Pair;

#[derive(Clone, Debug, Default)]
pub struct ParserMetadata {
    pub line: usize,
    pub column: usize,
    pub source_text: String,
}

impl ParserMetadata {
    pub fn new<R: Copy + Debug + Eq + Hash + Ord>(pair: &Pair<R>) -> Self {
        let (line, column) = pair.line_col();
        Self {
            line,
            column,
            source_text: pair.get_input()[pair.as_span().start()..pair.as_span().end()].to_string(),
        }
    }
}
