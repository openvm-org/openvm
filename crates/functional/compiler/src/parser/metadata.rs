use proc_macro2::Span;

#[derive(Clone, Debug, Default)]
pub struct ParserMetadata {
    pub source_text: Option<String>,
}

impl ParserMetadata {
    pub fn new(span: Span) -> Self {
        Self {
            source_text: span.source_text(),
        }
    }
}