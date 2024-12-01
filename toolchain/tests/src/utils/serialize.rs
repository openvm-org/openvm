pub fn serialize_u8_65_vec<S>(v: &Vec<[u8; 65]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(v.len() * 65))?;
    for arr in v {
        for &byte in arr {
            seq.serialize_element(&byte)?;
        }
    }
    seq.end()
}

pub fn serialize_u8_20_vec<S>(v: &Vec<[u8; 20]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(v.len() * 20))?;
    for arr in v {
        for &byte in arr {
            seq.serialize_element(&byte)?;
        }
    }
    seq.end()
}
