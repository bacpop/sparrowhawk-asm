//! Errors returned by the complete and component assembly APIs.

use crate::{graph_works::spelling::SpellError, preprocessing::PreprocessingError};
use std::{error::Error, fmt, io};

/// An invalid configuration, input failure, biological rejection, or output failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum AssemblyError {
    /// Settings cannot be used together or exceed the supported representation.
    InvalidOptions(String),
    /// Opening, parsing, or storing an input record failed.
    Input(io::Error),
    /// A k could not produce a useful assembly step.
    Preprocessing(PreprocessingError),
    /// A collapsed walk could not be recovered as a sequence.
    Spelling(SpellError),
    /// The caller's graph-output callback failed.
    GraphOutput(io::Error),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) => f.write_str(message),
            Self::Input(error) => write!(f, "input: {error}"),
            Self::Preprocessing(error) => error.fmt(f),
            Self::Spelling(error) => error.fmt(f),
            Self::GraphOutput(error) => write!(f, "graph output: {error}"),
        }
    }
}

impl Error for AssemblyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidOptions(_) => None,
            Self::Input(error) | Self::GraphOutput(error) => Some(error),
            Self::Preprocessing(error) => Some(error),
            Self::Spelling(error) => Some(error),
        }
    }
}

impl From<io::Error> for AssemblyError {
    fn from(error: io::Error) -> Self {
        Self::Input(error)
    }
}
impl From<PreprocessingError> for AssemblyError {
    fn from(error: PreprocessingError) -> Self {
        Self::Preprocessing(error)
    }
}
impl From<SpellError> for AssemblyError {
    fn from(error: SpellError) -> Self {
        Self::Spelling(error)
    }
}
