//! Where reads come from: files on disk or the browser's `File`s, which are handles, not the file data themselves. Either can be read from the start again,
//! since a recount at a looser floor, or the next k of a ladder, starts over.

/// Tuple for name and list of input files
#[cfg(not(target_family = "wasm"))]
pub(crate) type InputFastx = (String, Vec<String>);

/// One owned read: ASCII bases and optional PHRED+33 quality bytes of the same length.
/// FASTA records have no qualities. Non-ACGT bases break k-mers in normal storage;
/// two-bit read storage rejects them.
pub type OwnedRecord = (Vec<u8>, Option<Vec<u8>>); // nts, option<qualities>

/// `Send` natively, where a producer thread parses ahead of the counter; nothing in the browser, which
/// has no threads, and whose `File`s could not cross one anyway.
/// NOTE: the "Send" trait is the trait we need to transfer things between threads.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send> MaybeSend for T {}

/// This is the definition for wasm: here we cannot have Send, because javascript file handles cannot be shared across threads (I mean the "web_sys::File" handle, that is not the file's data itself)
#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T> MaybeSend for T {}

/// A repeatable source. Every pass must yield the same reads in the same order.
///
/// A pass owns its records. Natively its iterator must be `Send`, because parsing can run
/// on a scoped producer thread; the factory and source need not be `Send` or `Sync`.
/// Do not convert read errors to end-of-input: later passes may follow a quality peek,
/// a recount, or another k value.
pub trait ReadSource {
    /// One pass, with opening/parse failures carried explicitly.
    type Records<'a>: Iterator<Item = std::io::Result<OwnedRecord>> + MaybeSend
    where
        Self: 'a;
    /// Start a fresh pass; errors must not be treated as end-of-input.
    fn records(&self) -> std::io::Result<Self::Records<'_>>;
}

/// A repeatable custom parser factory. Wrap infallible iterator items with `.map(Ok)`.
///
/// ```
/// use sparrowhawk_asm::{IteratorSource, OwnedRecord, ReadSource};
/// let records: Vec<OwnedRecord> = vec![(b"ACGT".to_vec(), None)];
/// let source = IteratorSource::new(|| Ok(records.clone().into_iter().map(Ok)));
/// assert_eq!(source.records()?.collect::<std::io::Result<Vec<_>>>()?, records);
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct IteratorSource<F> {
    factory: F,
}
impl<F> IteratorSource<F> {
    /// Call `factory` once for each requested pass, including quality peeks.
    pub fn new(factory: F) -> Self {
        Self { factory }
    }
}
impl<F, I> ReadSource for IteratorSource<F>
where
    F: Fn() -> std::io::Result<I>,
    I: Iterator<Item = std::io::Result<OwnedRecord>> + MaybeSend,
{
    type Records<'a>
        = I
    where
        Self: 'a;
    fn records(&self) -> std::io::Result<I> {
        (self.factory)()
    }
}

/// Adapter for callers already using a needletail parser.
#[cfg(not(target_family = "wasm"))]
pub use super::io_utils::NeedletailIterator;

/// Native paths owned by the source. Files open lazily, one at a time, on each pass.
#[cfg(not(target_family = "wasm"))]
pub struct InputFiles {
    paths: Vec<std::path::PathBuf>,
}
#[cfg(not(target_family = "wasm"))]
impl InputFiles {
    /// Create a source without opening files. An empty path list is invalid.
    pub fn new<P: Into<std::path::PathBuf>>(
        paths: impl IntoIterator<Item = P>,
    ) -> std::io::Result<Self> {
        let paths: Vec<_> = paths.into_iter().map(Into::into).collect();
        if paths.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no input files",
            ));
        }
        Ok(Self { paths })
    }
}

/// Opaque iterator over native files; ends after the first opening or parsing failure.
#[cfg(not(target_family = "wasm"))]
pub struct FileRecords<'a> {
    paths: std::slice::Iter<'a, std::path::PathBuf>,
    current: Option<(NeedletailIterator, &'a std::path::Path)>,
    failed: bool,
}
#[cfg(not(target_family = "wasm"))]
impl ReadSource for InputFiles {
    type Records<'a> = FileRecords<'a>;
    fn records(&self) -> std::io::Result<FileRecords<'_>> {
        Ok(FileRecords {
            paths: self.paths.iter(),
            current: None,
            failed: false,
        })
    }
}
#[cfg(not(target_family = "wasm"))]
impl Iterator for FileRecords<'_> {
    type Item = std::io::Result<OwnedRecord>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if let Some((records, path)) = &mut self.current {
                if let Some(record) = records.next() {
                    return Some(record.map_err(|error| {
                        self.failed = true;
                        std::io::Error::new(error.kind(), format!("{}: {error}", path.display()))
                    }));
                }
                self.current = None;
            }
            let path = self.paths.next()?;
            let reader = std::fs::File::open(path).and_then(|file| {
                needletail::parse_fastx_reader(file)
                    .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
            });
            match reader {
                Ok(reader) => self.current = Some((NeedletailIterator::new(reader), path)),
                Err(error) => {
                    self.failed = true;
                    return Some(Err(std::io::Error::new(
                        error.kind(),
                        format!("{}: {error}", path.display()),
                    )));
                }
            }
        }
    }
}

/// Browser file handles, reopened lazily for recounts and later k values.
#[cfg(target_family = "wasm")]
pub struct BrowserFiles {
    files: Vec<web_sys::File>,
}
#[cfg(target_family = "wasm")]
impl BrowserFiles {
    /// Reject an empty file list; file contents are read only when a pass requests records.
    pub fn new(files: Vec<web_sys::File>) -> std::io::Result<Self> {
        if files.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no input files",
            ));
        }
        Ok(Self { files })
    }
}
/// Opaque browser parser iterator, retaining file context on failures.
#[cfg(target_family = "wasm")]
pub struct BrowserRecords<'a> {
    files: std::slice::Iter<'a, web_sys::File>,
    current: Option<(super::fastx_wasm::FastxRecords, String)>,
    failed: bool,
}
#[cfg(target_family = "wasm")]
impl ReadSource for BrowserFiles {
    type Records<'a> = BrowserRecords<'a>;
    fn records(&self) -> std::io::Result<BrowserRecords<'_>> {
        let total = self.files.iter().map(|file| file.size()).sum();
        PASS_PROGRESS.with(|p| p.set((0.0, total)));
        Ok(BrowserRecords {
            files: self.files.iter(),
            current: None,
            failed: false,
        })
    }
}
#[cfg(target_family = "wasm")]
impl Iterator for BrowserRecords<'_> {
    type Item = std::io::Result<OwnedRecord>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if let Some((records, name)) = &mut self.current {
                if let Some(record) = records.next() {
                    return Some(record.map_err(|error| {
                        self.failed = true;
                        std::io::Error::new(error.kind(), format!("{name}: {error}"))
                    }));
                }
                self.current = None;
            }
            let file = self.files.next()?;
            match super::fastx_wasm::FastxRecords::open(Counted(
                wasm_bindgen_file_reader::WebSysFile::new(file.clone()),
            )) {
                Ok(records) => self.current = Some((records, file.name())),
                Err(error) => {
                    self.failed = true;
                    return Some(Err(std::io::Error::new(
                        error.kind(),
                        format!("{}: {error}", file.name()),
                    )));
                }
            }
        }
    }
}

/// Reject malformed qualities before any counting or storage mutation.
pub(crate) fn validate_record(record: OwnedRecord) -> std::io::Result<OwnedRecord> {
    validate_quality(&record.0, record.1.as_deref())?;
    Ok(record)
}
pub(crate) fn validate_quality(seq: &[u8], qual: Option<&[u8]>) -> std::io::Result<()> {
    if qual.is_some_and(|q| q.len() != seq.len()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "sequence and quality lengths differ",
        ));
    }
    Ok(())
}

// This is quite magic, but it is interesting: this is a way of having a shared variable that lies in the current thread, given that we have the two-buffer system. Those generate, IIUC, scopes that do not speak to each other in any way. This thread_local! macro creates essentially a, thanks to the "static", global variable for all the thread. Because it's static for globalness, we need std::cell::Cell for having mutability in its interior, as static mut is unsafe and forbidden in recent Rust versions. All a bit quite magic, but I did something similar when I didn't know in Python how to pass info to shared threads, now that I remember. We use this for showing progress on the web.

#[cfg(target_family = "wasm")]
thread_local! {
    /// Bytes the current pass over the page's files has read, and their combined size. Compressed
    /// bytes for gzipped files, which is still how far through them the pass is.
    static PASS_PROGRESS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((0.0, 0.0)) };
}

/// How far through the page's files the current pass is, in percent, for the page's progress line.
#[cfg(target_family = "wasm")]
pub(crate) fn pass_percent() -> Option<u32> {
    let (read, total) = PASS_PROGRESS.with(std::cell::Cell::get);
    (total > 0.0).then(|| (100.0 * read / total).min(100.0) as u32)
}

/// WASM: A file whose reads add to the pass's progress.
#[cfg(target_family = "wasm")]
struct Counted<R>(R);

#[cfg(target_family = "wasm")]
impl<R: std::io::Read> std::io::Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.0.read(buf)?;
        PASS_PROGRESS.with(|progress| {
            let (read, total) = progress.get();
            progress.set((read + n as f64, total));
        });
        Ok(n)
    }
}
