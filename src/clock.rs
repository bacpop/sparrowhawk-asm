//! A monotonic clock on both targets. `std::time::Instant::now` panics in the browser, so there it reads
//! `performance.now()`, which every Worker has.

#[cfg(not(target_family = "wasm"))]
pub use std::time::Instant;

#[cfg(target_family = "wasm")]
pub use browser::Instant;

#[cfg(target_family = "wasm")]
mod browser {
    use std::time::Duration;

    /// Milliseconds since the worker started.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        /// Now.
        pub fn now() -> Self {
            Self(crate::performance_now())
        }

        /// Time from `earlier` to this one, zero if `earlier` is later.
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1000.0)
        }

        /// Time since this one.
        pub fn elapsed(&self) -> Duration {
            Self::now().duration_since(*self)
        }
    }
}
