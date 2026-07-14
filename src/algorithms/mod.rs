//! Various algorithms used to create, modify and collapse graphs
pub mod collapser;
pub mod corrector;
/// Using a larger k as evidence to judge the branches of a smaller-k graph. Native only: it needs
/// `PreprocessedK`, and the wasm build has no multi-k path.
#[cfg(not(target_family = "wasm"))]
pub mod multik;
pub mod shrinker;
