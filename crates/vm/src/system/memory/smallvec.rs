use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};
use smallvec::{Array, SmallVec};

pub struct SerdeSmallVec<A: Array>(SmallVec<A>);

impl<A: Array> Debug for SerdeSmallVec<A>
where
    A::Item: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<A: Array> Clone for SerdeSmallVec<A>
where
    A::Item: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<A: Array> From<SmallVec<A>> for SerdeSmallVec<A> {
    fn from(smallvec: SmallVec<A>) -> Self {
        Self(smallvec)
    }
}

impl<A: Array> From<SerdeSmallVec<A>> for SmallVec<A> {
    fn from(wrapper: SerdeSmallVec<A>) -> Self {
        wrapper.0
    }
}

// Implement Deref so SerdeSmallVec behaves exactly like SmallVec
impl<A: Array> Deref for SerdeSmallVec<A> {
    type Target = SmallVec<A>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Implement DerefMut to allow mutable access
impl<A: Array> DerefMut for SerdeSmallVec<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Implement Serialize and Deserialize only when needed
impl<A: Array + Serialize> Serialize for SerdeSmallVec<A>
where
    A::Item: Serialize, // Ensure the item type is serializable
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.as_slice().serialize(serializer)
    }
}

impl<'de, A: Array + Deserialize<'de>> Deserialize<'de> for SerdeSmallVec<A>
where
    A::Item: Deserialize<'de>, // Ensure the item type is deserializable
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<A::Item>::deserialize(deserializer)?;
        Ok(Self(SmallVec::from_vec(vec)))
    }
}
