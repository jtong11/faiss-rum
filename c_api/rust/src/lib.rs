use libloading::{Library, Symbol};
use std::env;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{Display, Formatter};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

type FaissIdx = i64;

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    InnerProduct = 0,
    L2 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FaissVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl FaissVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn parse(version: &str) -> Option<Self> {
        let mut parts = version.split('.');
        let major = parts.next()?;
        let minor = parts.next()?;
        let patch = parts.next()?;

        let parse_component = |part: &str| -> Option<u32> {
            let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                return None;
            }
            digits.parse().ok()
        };

        Some(Self::new(
            parse_component(major)?,
            parse_component(minor)?,
            parse_component(patch)?,
        ))
    }
}

impl Display for FaissVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

const MIN_IVF_RABITQ_VERSION: FaissVersion = FaissVersion::new(1, 11, 0);

#[derive(Debug)]
pub enum FaissError {
    LibraryLoad(String),
    Api(String),
    NullPointer(&'static str),
    InvalidArgument(String),
    DimensionMismatch {
        expected_multiple_of: usize,
        actual_len: usize,
    },
    UnsupportedIndexType(&'static str),
    VersionParse(String),
    VersionTooOld {
        found: String,
        required: FaissVersion,
    },
}

impl Display for FaissError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryLoad(message) => write!(f, "{message}"),
            Self::Api(message) => write!(f, "Faiss API call failed: {message}"),
            Self::NullPointer(message) => write!(f, "{message}"),
            Self::InvalidArgument(message) => write!(f, "{message}"),
            Self::DimensionMismatch {
                expected_multiple_of,
                actual_len,
            } => write!(
                f,
                "input length ({actual_len}) is not a multiple of index dimension ({expected_multiple_of})"
            ),
            Self::UnsupportedIndexType(message) => write!(f, "{message}"),
            Self::VersionParse(version) => {
                write!(f, "failed to parse Faiss version string: {version}")
            }
            Self::VersionTooOld { found, required } => {
                write!(
                    f,
                    "Faiss version {found} does not support IVF-RaBitQ; requires >= {required}"
                )
            }
        }
    }
}

impl Error for FaissError {}

#[repr(C)]
struct FaissIndexOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct FaissIndexIVFOpaque {
    _private: [u8; 0],
}

type FnGetVersion = unsafe extern "C" fn() -> *const c_char;
type FnGetLastError = unsafe extern "C" fn() -> *const c_char;
type FnIndexFactory = unsafe extern "C" fn(
    *mut *mut FaissIndexOpaque,
    c_int,
    *const c_char,
    c_int,
) -> c_int;
type FnIndexFree = unsafe extern "C" fn(*mut FaissIndexOpaque);
type FnIndexD = unsafe extern "C" fn(*const FaissIndexOpaque) -> c_int;
type FnIndexNTotal = unsafe extern "C" fn(*const FaissIndexOpaque) -> FaissIdx;
type FnIndexTrain = unsafe extern "C" fn(*mut FaissIndexOpaque, FaissIdx, *const f32) -> c_int;
type FnIndexAdd = unsafe extern "C" fn(*mut FaissIndexOpaque, FaissIdx, *const f32) -> c_int;
type FnIndexSearch = unsafe extern "C" fn(
    *const FaissIndexOpaque,
    FaissIdx,
    *const f32,
    FaissIdx,
    *mut f32,
    *mut FaissIdx,
) -> c_int;
type FnIndexIVFCast = unsafe extern "C" fn(*mut FaissIndexOpaque) -> *mut FaissIndexIVFOpaque;
type FnIndexIVFSetNProbe = unsafe extern "C" fn(*mut FaissIndexIVFOpaque, usize);

struct FaissApi {
    _lib: Library,
    get_version: FnGetVersion,
    get_last_error: FnGetLastError,
    index_factory: FnIndexFactory,
    index_free: FnIndexFree,
    index_d: FnIndexD,
    index_ntotal: FnIndexNTotal,
    index_train: FnIndexTrain,
    index_add: FnIndexAdd,
    index_search: FnIndexSearch,
    index_ivf_cast: FnIndexIVFCast,
    index_ivf_set_nprobe: FnIndexIVFSetNProbe,
}

impl FaissApi {
    fn load() -> Result<Arc<Self>, FaissError> {
        let mut candidates = Vec::new();
        if let Ok(path) = env::var("FAISS_C_LIB_PATH") {
            candidates.push(path);
        }
        candidates.push("/workspace/build/c_api/libfaiss_c.so".to_owned());
        candidates.push("/workspace/build/libfaiss_c.so".to_owned());
        candidates.push("libfaiss_c.so".to_owned());
        candidates.push("faiss_c".to_owned());

        let mut errors = Vec::new();
        for candidate in candidates {
            if candidate.contains('/') && !Path::new(&candidate).exists() {
                continue;
            }

            match unsafe { Library::new(&candidate) } {
                Ok(lib) => {
                    let api = unsafe { Self::from_library(lib) }?;
                    return Ok(Arc::new(api));
                }
                Err(err) => errors.push(format!("{candidate}: {err}")),
            }
        }

        Err(FaissError::LibraryLoad(format!(
            "failed to load libfaiss_c. Set FAISS_C_LIB_PATH to the shared library path. Attempts: {}",
            errors.join(" | ")
        )))
    }

    unsafe fn from_library(lib: Library) -> Result<Self, FaissError> {
        unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, FaissError> {
            let symbol: Symbol<'_, T> = lib.get(name).map_err(|err| {
                FaissError::LibraryLoad(format!(
                    "missing symbol {}: {err}",
                    String::from_utf8_lossy(name)
                ))
            })?;
            Ok(*symbol)
        }

        Ok(Self {
            get_version: load_symbol(&lib, b"faiss_get_version\0")?,
            get_last_error: load_symbol(&lib, b"faiss_get_last_error\0")?,
            index_factory: load_symbol(&lib, b"faiss_index_factory\0")?,
            index_free: load_symbol(&lib, b"faiss_Index_free\0")?,
            index_d: load_symbol(&lib, b"faiss_Index_d\0")?,
            index_ntotal: load_symbol(&lib, b"faiss_Index_ntotal\0")?,
            index_train: load_symbol(&lib, b"faiss_Index_train\0")?,
            index_add: load_symbol(&lib, b"faiss_Index_add\0")?,
            index_search: load_symbol(&lib, b"faiss_Index_search\0")?,
            index_ivf_cast: load_symbol(&lib, b"faiss_IndexIVF_cast\0")?,
            index_ivf_set_nprobe: load_symbol(&lib, b"faiss_IndexIVF_set_nprobe\0")?,
            _lib: lib,
        })
    }

    fn last_error_message(&self) -> String {
        unsafe {
            let msg_ptr = (self.get_last_error)();
            if msg_ptr.is_null() {
                "Faiss call failed without an error message".to_owned()
            } else {
                CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn check_error(&self, code: c_int) -> Result<(), FaissError> {
        if code == 0 {
            Ok(())
        } else {
            Err(FaissError::Api(self.last_error_message()))
        }
    }

    fn version_string(&self) -> Result<String, FaissError> {
        unsafe {
            let version_ptr = (self.get_version)();
            if version_ptr.is_null() {
                return Err(FaissError::NullPointer(
                    "faiss_get_version returned a null pointer",
                ));
            }
            Ok(CStr::from_ptr(version_ptr).to_string_lossy().into_owned())
        }
    }
}

pub fn faiss_version_string() -> Result<String, FaissError> {
    let api = FaissApi::load()?;
    api.version_string()
}

pub fn faiss_version() -> Result<FaissVersion, FaissError> {
    let version = faiss_version_string()?;
    FaissVersion::parse(&version).ok_or(FaissError::VersionParse(version))
}

fn ensure_ivf_rabitq_support(api: &FaissApi) -> Result<(), FaissError> {
    let version_string = api.version_string()?;
    let parsed = FaissVersion::parse(&version_string)
        .ok_or_else(|| FaissError::VersionParse(version_string.clone()))?;

    if parsed < MIN_IVF_RABITQ_VERSION {
        return Err(FaissError::VersionTooOld {
            found: version_string,
            required: MIN_IVF_RABITQ_VERSION,
        });
    }
    Ok(())
}

struct FaissIndexHandle {
    api: Arc<FaissApi>,
    ptr: *mut FaissIndexOpaque,
    dimension: usize,
}

impl FaissIndexHandle {
    fn new_from_factory(
        dimension: usize,
        factory: &str,
        metric: MetricType,
    ) -> Result<Self, FaissError> {
        if dimension == 0 {
            return Err(FaissError::InvalidArgument(
                "dimension must be greater than 0".to_owned(),
            ));
        }
        if dimension > c_int::MAX as usize {
            return Err(FaissError::InvalidArgument(format!(
                "dimension {dimension} exceeds c_int::MAX"
            )));
        }

        let api = FaissApi::load()?;
        let description =
            CString::new(factory).expect("factory string does not contain interior NUL");

        let mut index_ptr: *mut FaissIndexOpaque = ptr::null_mut();
        unsafe {
            api.check_error((api.index_factory)(
                &mut index_ptr,
                dimension as c_int,
                description.as_ptr(),
                metric as c_int,
            ))?;
        }

        if index_ptr.is_null() {
            return Err(FaissError::NullPointer(
                "faiss_index_factory succeeded but returned a null index pointer",
            ));
        }

        let index_dimension = unsafe { (api.index_d)(index_ptr) };
        if index_dimension <= 0 {
            return Err(FaissError::InvalidArgument(format!(
                "factory returned invalid index dimension: {index_dimension}"
            )));
        }

        Ok(Self {
            api,
            ptr: index_ptr,
            dimension: index_dimension as usize,
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn ntotal(&self) -> usize {
        unsafe { (self.api.index_ntotal)(self.ptr) as usize }
    }

    fn set_nprobe(&mut self, nprobe: usize) -> Result<(), FaissError> {
        if nprobe == 0 {
            return Err(FaissError::InvalidArgument(
                "nprobe must be greater than 0".to_owned(),
            ));
        }

        unsafe {
            let ivf_ptr = (self.api.index_ivf_cast)(self.ptr);
            if ivf_ptr.is_null() {
                return Err(FaissError::UnsupportedIndexType(
                    "factory result is not an IVF index",
                ));
            }
            (self.api.index_ivf_set_nprobe)(ivf_ptr, nprobe);
        }
        Ok(())
    }

    fn train(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        let n = self.validate_vector_matrix(vectors)?;
        unsafe { self.api.check_error((self.api.index_train)(self.ptr, n, vectors.as_ptr())) }
    }

    fn add(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        let n = self.validate_vector_matrix(vectors)?;
        unsafe { self.api.check_error((self.api.index_add)(self.ptr, n, vectors.as_ptr())) }
    }

    fn search(&self, queries: &[f32], k: usize) -> Result<SearchResult, FaissError> {
        if k == 0 {
            return Err(FaissError::InvalidArgument(
                "k must be greater than 0".to_owned(),
            ));
        }
        if k > i64::MAX as usize {
            return Err(FaissError::InvalidArgument(format!(
                "k {k} exceeds i64::MAX"
            )));
        }

        let nq = self.validate_vector_matrix(queries)?;
        let nq_usize = nq as usize;
        let mut distances = vec![0.0_f32; nq_usize * k];
        let mut labels = vec![-1_i64; nq_usize * k];

        unsafe {
            self.api.check_error((self.api.index_search)(
                self.ptr,
                nq,
                queries.as_ptr(),
                k as FaissIdx,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            ))?;
        }

        Ok(SearchResult {
            distances,
            labels,
            k,
        })
    }

    fn validate_vector_matrix(&self, values: &[f32]) -> Result<FaissIdx, FaissError> {
        if values.is_empty() {
            return Err(FaissError::InvalidArgument(
                "input vector buffer must not be empty".to_owned(),
            ));
        }
        if values.len() % self.dimension != 0 {
            return Err(FaissError::DimensionMismatch {
                expected_multiple_of: self.dimension,
                actual_len: values.len(),
            });
        }

        let n = values.len() / self.dimension;
        if n > i64::MAX as usize {
            return Err(FaissError::InvalidArgument(format!(
                "number of vectors {n} exceeds i64::MAX"
            )));
        }
        Ok(n as FaissIdx)
    }
}

impl Drop for FaissIndexHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.api.index_free)(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct IvfRaBitQIndex {
    inner: FaissIndexHandle,
}

impl IvfRaBitQIndex {
    pub fn new(dimension: usize, nlist: usize, metric: MetricType) -> Result<Self, FaissError> {
        if nlist == 0 {
            return Err(FaissError::InvalidArgument(
                "nlist must be greater than 0".to_owned(),
            ));
        }

        let api = FaissApi::load()?;
        ensure_ivf_rabitq_support(&api)?;
        drop(api);

        Ok(Self {
            inner: FaissIndexHandle::new_from_factory(
                dimension,
                &format!("IVF{nlist},RaBitQ"),
                metric,
            )?,
        })
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    pub fn ntotal(&self) -> usize {
        self.inner.ntotal()
    }

    pub fn set_nprobe(&mut self, nprobe: usize) -> Result<(), FaissError> {
        self.inner.set_nprobe(nprobe)
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.train(vectors)
    }

    pub fn add(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.add(vectors)
    }

    pub fn search(&self, queries: &[f32], k: usize) -> Result<SearchResult, FaissError> {
        self.inner.search(queries, k)
    }
}

pub struct IvfSq8Index {
    inner: FaissIndexHandle,
}

impl IvfSq8Index {
    pub fn new(dimension: usize, nlist: usize, metric: MetricType) -> Result<Self, FaissError> {
        if nlist == 0 {
            return Err(FaissError::InvalidArgument(
                "nlist must be greater than 0".to_owned(),
            ));
        }

        Ok(Self {
            inner: FaissIndexHandle::new_from_factory(
                dimension,
                &format!("IVF{nlist},SQ8"),
                metric,
            )?,
        })
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    pub fn ntotal(&self) -> usize {
        self.inner.ntotal()
    }

    pub fn set_nprobe(&mut self, nprobe: usize) -> Result<(), FaissError> {
        self.inner.set_nprobe(nprobe)
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.train(vectors)
    }

    pub fn add(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.add(vectors)
    }

    pub fn search(&self, queries: &[f32], k: usize) -> Result<SearchResult, FaissError> {
        self.inner.search(queries, k)
    }
}

pub struct HnswIndex {
    inner: FaissIndexHandle,
}

impl HnswIndex {
    pub fn new(dimension: usize, m: usize, metric: MetricType) -> Result<Self, FaissError> {
        if m == 0 {
            return Err(FaissError::InvalidArgument(
                "hnsw M must be greater than 0".to_owned(),
            ));
        }

        Ok(Self {
            inner: FaissIndexHandle::new_from_factory(dimension, &format!("HNSW{m}"), metric)?,
        })
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    pub fn ntotal(&self) -> usize {
        self.inner.ntotal()
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.train(vectors)
    }

    pub fn add(&mut self, vectors: &[f32]) -> Result<(), FaissError> {
        self.inner.add(vectors)
    }

    pub fn search(&self, queries: &[f32], k: usize) -> Result<SearchResult, FaissError> {
        self.inner.search(queries, k)
    }
}

pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<i64>,
    pub k: usize,
}

impl SearchResult {
    pub fn labels_for_query(&self, query_idx: usize) -> &[i64] {
        let start = query_idx * self.k;
        let end = start + self.k;
        &self.labels[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_vectors(n: usize, d: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(n * d);
        let mut state = 0x1234_5678_9abc_def0_u64;
        for i in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random = ((state >> 33) as f32) / ((1_u32 << 31) as f32);
                values.push(random + i as f32 * 1e-3 + j as f32 * 1e-4);
            }
        }
        values
    }

    fn skip_if_library_missing(err: &FaissError) -> bool {
        matches!(err, FaissError::LibraryLoad(_))
    }

    #[test]
    fn parse_version_string() {
        let version = FaissVersion::parse("1.11.0-dev").expect("version should parse");
        assert_eq!(version, FaissVersion::new(1, 11, 0));
    }

    #[test]
    fn faiss_version_is_new_enough_for_ivf_rabitq() -> Result<(), Box<dyn Error>> {
        let version = match faiss_version() {
            Ok(v) => v,
            Err(err) if skip_if_library_missing(&err) => {
                eprintln!("skipping version check test: {err}");
                return Ok(());
            }
            Err(err) => return Err(Box::new(err)),
        };

        assert!(
            version >= MIN_IVF_RABITQ_VERSION,
            "found Faiss {version}, requires >= {MIN_IVF_RABITQ_VERSION}"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_rabitq_configuration() {
        let result = IvfRaBitQIndex::new(8, 0, MetricType::L2);
        assert!(matches!(result, Err(FaissError::InvalidArgument(_))));
    }

    #[test]
    fn train_add_and_search_ivf_rabitq() -> Result<(), Box<dyn Error>> {
        let d = 16;
        let nlist = 8;
        let nb = 512;
        let k = 50;

        let mut index = match IvfRaBitQIndex::new(d, nlist, MetricType::L2) {
            Ok(index) => index,
            Err(err) if skip_if_library_missing(&err) => {
                eprintln!("skipping IVF-RaBitQ integration test: {err}");
                return Ok(());
            }
            Err(err) => return Err(Box::new(err)),
        };

        let xb = synthetic_vectors(nb, d);
        index.train(&xb)?;
        index.add(&xb)?;
        index.set_nprobe(nlist)?;

        assert_eq!(index.ntotal(), nb);

        let query_ids = [0_usize, 64_usize, 255_usize];
        let mut queries = Vec::with_capacity(query_ids.len() * d);
        for id in query_ids {
            let start = id * d;
            queries.extend_from_slice(&xb[start..start + d]);
        }

        let results = index.search(&queries, k)?;
        assert_eq!(results.labels.len(), query_ids.len() * k);
        assert_eq!(results.distances.len(), query_ids.len() * k);
        assert!(results.distances.iter().all(|distance| distance.is_finite()));

        for (query_row, expected_id) in [0_i64, 64_i64, 255_i64].into_iter().enumerate() {
            assert!(
                results.labels_for_query(query_row).contains(&expected_id),
                "expected id {expected_id} to appear in top-{k} for query row {query_row}"
            );
        }

        Ok(())
    }

    #[test]
    fn rejects_invalid_ivf_sq8_configuration() {
        let result = IvfSq8Index::new(8, 0, MetricType::L2);
        assert!(matches!(result, Err(FaissError::InvalidArgument(_))));
    }

    #[test]
    fn train_add_and_search_ivf_sq8() -> Result<(), Box<dyn Error>> {
        let d = 16;
        let nlist = 8;
        let nb = 512;
        let k = 50;

        let mut index = match IvfSq8Index::new(d, nlist, MetricType::L2) {
            Ok(index) => index,
            Err(err) if skip_if_library_missing(&err) => {
                eprintln!("skipping IVF-SQ8 integration test: {err}");
                return Ok(());
            }
            Err(err) => return Err(Box::new(err)),
        };

        let xb = synthetic_vectors(nb, d);
        index.train(&xb)?;
        index.add(&xb)?;
        index.set_nprobe(nlist)?;

        assert_eq!(index.ntotal(), nb);

        let query_ids = [0_usize, 64_usize, 255_usize];
        let mut queries = Vec::with_capacity(query_ids.len() * d);
        for id in query_ids {
            let start = id * d;
            queries.extend_from_slice(&xb[start..start + d]);
        }

        let results = index.search(&queries, k)?;
        assert_eq!(results.labels.len(), query_ids.len() * k);
        assert_eq!(results.distances.len(), query_ids.len() * k);
        assert!(results.distances.iter().all(|distance| distance.is_finite()));

        for (query_row, expected_id) in [0_i64, 64_i64, 255_i64].into_iter().enumerate() {
            assert!(
                results.labels_for_query(query_row).contains(&expected_id),
                "expected id {expected_id} to appear in top-{k} for query row {query_row}"
            );
        }

        Ok(())
    }

    #[test]
    fn rejects_invalid_hnsw_configuration() {
        let result = HnswIndex::new(8, 0, MetricType::L2);
        assert!(matches!(result, Err(FaissError::InvalidArgument(_))));
    }

    #[test]
    fn add_and_search_hnsw() -> Result<(), Box<dyn Error>> {
        let d = 16;
        let m = 32;
        let nb = 512;
        let k = 64;

        let mut index = match HnswIndex::new(d, m, MetricType::L2) {
            Ok(index) => index,
            Err(err) if skip_if_library_missing(&err) => {
                eprintln!("skipping HNSW integration test: {err}");
                return Ok(());
            }
            Err(err) => return Err(Box::new(err)),
        };

        let xb = synthetic_vectors(nb, d);
        index.add(&xb)?;

        assert_eq!(index.ntotal(), nb);

        let query_ids = [1_usize, 64_usize, 255_usize];
        let mut queries = Vec::with_capacity(query_ids.len() * d);
        for id in query_ids {
            let start = id * d;
            queries.extend_from_slice(&xb[start..start + d]);
        }

        let results = index.search(&queries, k)?;
        assert_eq!(results.labels.len(), query_ids.len() * k);
        assert_eq!(results.distances.len(), query_ids.len() * k);
        assert!(results.distances.iter().all(|distance| distance.is_finite()));

        for (query_row, expected_id) in [1_i64, 64_i64, 255_i64].into_iter().enumerate() {
            assert!(
                results.labels_for_query(query_row).contains(&expected_id),
                "expected id {expected_id} to appear in top-{k} for query row {query_row}"
            );
        }

        Ok(())
    }
}

