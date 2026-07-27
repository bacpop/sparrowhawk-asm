//! Various algorithms used to create, modify and collapse graphs
pub mod collapser;
pub mod corrector;
/// Using a larger k as evidence to judge the branches of a smaller-k graph. Native only: it needs
/// `PreprocessedK`, and the wasm build has no multi-k path.
#[cfg(not(target_family = "wasm"))]
pub mod multik;
pub mod shrinker;
/// Finding superbubbles — forks with three or more branches, multi-unitig branches, or nesting — and
/// asking the multi-k oracle about them. Native only, for the same reason as `multik`: it needs the
/// evidence graph.
#[cfg(not(target_family = "wasm"))]
pub mod superbubble;
