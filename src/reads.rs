//! Where reads come from: files on disk or the browser's `File`s, which are handles, not the file data themselves. Either can be read from the start again,
//! since a recount at a looser floor, or the next k of a ladder, starts over.

/// One read: its bases and, from FASTQ, its quality bytes.
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

/// Reads that can be read from the start as often as a run needs.
pub trait ReadSource {
    /// One pass over every read, in file order.
    type Records<'a>: Iterator<Item = OwnedRecord> + MaybeSend
    where
        Self: 'a;

    /// Start a pass.
    fn records(&self) -> Self::Records<'_>;
}

/// For NATIVE: Files on disk, opened afresh for every pass. A single k reads them again for its recount; a multi-k run reads them once, packing them into a `ReadStore` that its later k values replay. This uses needletail at the end of the day, because we're running on native.
#[cfg(not(target_family = "wasm"))]
pub struct InputFiles<'a>(pub &'a [crate::preprocessing::InputFastx]);

#[cfg(not(target_family = "wasm"))]
impl ReadSource for InputFiles<'_> {
    type Records<'a>
        = std::iter::Flatten<std::vec::IntoIter<crate::io_utils::NeedletailIterator>>
    where
        Self: 'a;

    fn records(&self) -> Self::Records<'_> {
        self.0
            .iter()
            .flat_map(|(_, files)| files)
            .map(|file| {
                let reader = needletail::parse_fastx_file(file)
                    .unwrap_or_else(|_| panic!("Invalid path/file: {file}"));
                crate::io_utils::NeedletailIterator::new(reader)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
    }
}

/// WASM: the browser's `File`s, opened afresh for every pass. The browser keeps them readable while the page holds them, so nothing is copied into memory. Here we use the fastx_wasm.rs structs and functions.
#[cfg(target_family = "wasm")]
pub struct BrowserFiles(pub Vec<web_sys::File>);

#[cfg(target_family = "wasm")]
impl ReadSource for BrowserFiles {
    type Records<'a>
        = std::iter::Flatten<std::vec::IntoIter<crate::fastx_wasm::FastxRecords>>
    where
        Self: 'a;

    /// Also restarts the progress count, against the files' combined size.
    fn records(&self) -> Self::Records<'_> {
        let total: f64 = self.0.iter().map(|file| file.size()).sum();
        PASS_PROGRESS.with(|progress| progress.set((0.0, total)));
        self.0
            .iter()
            .map(|file| {
                crate::fastx_wasm::FastxRecords::open(Counted(
                    wasm_bindgen_file_reader::WebSysFile::new(file.clone()),
                ))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
    }
}

// This is quite magic, but it is interesting: this is a way of having a shared variable that lies in the current thread, given that we have the two-buffer system. Those generate, IIUC, scopes that do not speak to each other in any way. This thread_local! macro creates essentially a, thanks to the "static", global variable for all the thread. Because it's static for globalness, we need std:cell:Cell for having mutability in its interior, as static mut is unsafe and forbidden in recent Rust versions. All a bit quite magic, but I did something similar when I didn't know in Python how to pass info to shared threads, now that I remember. We use this for showing progress on the web.

#[cfg(target_family = "wasm")]
thread_local! {
    /// Bytes the current pass over the page's files has read, and their combined size. Compressed
    /// bytes for gzipped files, which is still how far through them the pass is.
    static PASS_PROGRESS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((0.0, 0.0)) };
}

/// How far through the page's files the current pass is, in percent, for the page's progress line.
#[cfg(target_family = "wasm")]
pub fn pass_percent() -> Option<u32> {
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
