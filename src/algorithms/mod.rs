//! Various algorithms used to create, modify and collapse graphs
pub mod collapser;
pub mod corrector;
/// Minia-style removal of short low-coverage alternative paths and erroneous connections
#[cfg(not(target_family = "wasm"))]
pub mod path_correction;
pub mod shrinker;
