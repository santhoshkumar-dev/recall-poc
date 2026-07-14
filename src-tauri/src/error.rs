use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecallError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Serialize for RecallError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl From<String> for RecallError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

impl From<&str> for RecallError {
    fn from(value: &str) -> Self {
        Self::Message(value.to_owned())
    }
}

pub type Result<T> = std::result::Result<T, RecallError>;
