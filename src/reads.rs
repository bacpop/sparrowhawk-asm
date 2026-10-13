//! Where reads come from: files on disk or the page's `File`s. Either can be read from the start again,
//! since a recount at a looser floor, or the next k of a ladder, starts over.

/// One read: its bases and, from FASTQ, its quality bytes.
pub type OwnedRecord = (Vec<u8>, Option<Vec<u8>>);

/// `Send` natively, where a producer thread parses ahead of the counter; nothing in the browser, which
/// has no threads, and whose `File`s could not cross one anyway.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send> MaybeSend for T {}

/// This is the definition for wasm: here we cannot have Send, because javascript file handles cannot be shared across threads
/// (I mean the "web_sys::File" object)
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

/// Files on disk, opened afresh for every pass. A single k reads them again for its recount; a
/// multi-k run reads them once, packing them into a `ReadStore` that its later k values replay.
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

/// The page's `File`s, opened afresh for every pass. The browser keeps them readable while the page
/// holds them, so nothing is copied into memory.
#[cfg(target_family = "wasm")]
pub struct BrowserFiles(pub Vec<web_sys::File>);

#[cfg(target_family = "wasm")]
impl ReadSource for BrowserFiles {
    type Records<'a>
        = std::iter::Flatten<std::vec::IntoIter<crate::fastx_wasm::FastxRecords>>
    where
        Self: 'a;

    fn records(&self) -> Self::Records<'_> {
        self.0
            .iter()
            .map(|file| {
                crate::fastx_wasm::FastxRecords::open(wasm_bindgen_file_reader::WebSysFile::new(
                    file.clone(),
                ))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
    }
}
