//! Various algorithms used to create, modify and collapse graphs
pub mod collapser;
pub mod corrector;
/// Minia-style removal of short low-coverage alternative paths and erroneous connections
#[cfg(not(target_family = "wasm"))]
pub mod path_correction;
#[cfg(not(target_family = "wasm"))]
mod repeat_coverage;
/// Native repeat-motif detection and extraction before graph collapse.
#[cfg(not(target_family = "wasm"))]
pub(crate) mod repeat_recovery;
#[cfg(not(target_family = "wasm"))]
mod repeat_surgery;
pub mod shrinker;
#[cfg(not(target_family = "wasm"))]
mod superbubble;
