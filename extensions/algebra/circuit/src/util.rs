use num_bigint_dig::BigUint;
use serde::Deserialize;

pub(crate) fn deserialize_vec_biguint_from_str<'de, D>(
    deserializer: D,
) -> Result<Vec<BigUint>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<String> = Deserialize::deserialize(deserializer)?;
    let res = v.into_iter().map(|s| s.parse()).collect::<Vec<_>>();
    if res.iter().any(|x| x.is_err()) {
        return Err(serde::de::Error::custom("Failed to parse BigUint"));
    }
    Ok(res.into_iter().map(|x| x.unwrap()).collect())
}
