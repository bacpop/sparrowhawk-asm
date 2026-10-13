//! One logger for both targets, on the `log` crate alone: natively each line goes to stdout and to the
//! run's log file, in the browser to the console. Every line carries the time since logging started,
//! from [`crate::clock::Instant`], so the same lines time a run on both.

use std::sync::OnceLock;

use crate::clock::Instant;

struct Logger {
    start: Instant,
    /// The run's log file, appended to as fern did.
    #[cfg(not(target_family = "wasm"))]
    file: Option<std::sync::Mutex<std::io::LineWriter<std::fs::File>>>,
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "[{:>10.3} s {} {}] {}",
            self.start.elapsed().as_secs_f64(),
            record.level(),
            record.target(),
            record.args()
        );
        self.emit(record.level(), &line);
    }

    /// Lines are flushed as they are written: `println!` and `LineWriter` are line-buffered.
    fn flush(&self) {}
}

#[cfg(not(target_family = "wasm"))]
impl Logger {
    fn emit(&self, _level: log::Level, line: &str) {
        use std::io::Write;
        println!("{line}");
        if let Some(Ok(mut file)) = self.file.as_ref().map(|file| file.lock()) {
            let _ = writeln!(file, "{line}");
        }
    }
}

#[cfg(target_family = "wasm")]
impl Logger {
    fn emit(&self, level: log::Level, line: &str) {
        match level {
            log::Level::Error => crate::console_error(line),
            log::Level::Warn => crate::console_warn(line),
            _ => crate::console_log(line),
        }
    }
}

/// Log natively at `level`, to stdout and appended to `log_file`. A log file that cannot be opened
/// costs only that copy, with a warning on stderr.
#[cfg(not(target_family = "wasm"))]
pub fn init(level: log::LevelFilter, log_file: &std::path::Path) {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
        .map_err(|error| {
            eprintln!(
                "warning: cannot write the log file {}: {error}",
                log_file.display()
            )
        })
        .ok()
        .map(|file| std::sync::Mutex::new(std::io::LineWriter::new(file)));
    install(level, || Logger {
        start: Instant::now(),
        file,
    });
}

/// Log to the browser console at `level`.
#[cfg(target_family = "wasm")]
pub fn init(level: log::LevelFilter) {
    install(level, || Logger {
        start: Instant::now(),
    });
}

/// The first call sets the logger. A later one (a later page helper in the same worker) keeps that
/// logger and its start, and changes only the level.
fn install(level: log::LevelFilter, make: impl FnOnce() -> Logger) {
    let _ = log::set_logger(LOGGER.get_or_init(make));
    log::set_max_level(level);
}
