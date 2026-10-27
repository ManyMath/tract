//! Portable C ABI for the `tract` package's neural-network inference.
//!
//! One crate, two targets, one ABI:
//! - native (`cdylib`/`staticlib`) called from Dart over `dart:ffi`;
//! - `wasm32-unknown-unknown` called from Dart or JavaScript over
//!   `WebAssembly.instantiate`, with no `wasm-bindgen` glue.
//!
//! The Dart `FfiBackend` calls the model/value functions (`tract_onnx_model_load_*`,
//! `tract_nnef_model_load_*`, `tract_value_*`, `tract_model_run`). In addition,
//! WebAssembly and portable buffer callers can stage linear-memory buffers via
//! [`tract_alloc`]/[`tract_free`], load a model once with [`tract_model_load`]
//! (or [`tract_model_load_with_options`]) and call [`tract_model_run_f32`] /
//! [`tract_model_run_u8`] / [`tract_infer_28x28`] on the handle, inspecting
//! results via [`tract_buffer_status`], [`tract_buffer_data`],
//! [`tract_buffer_len`], and [`tract_buffer_free`]. [`tract_infer_bytes`] is the
//! one-shot variant that takes the model bytes on every call.
//!
//! Errors are reported both out-of-band via [`tract_last_error`] (and
//! [`tract_last_error_len`]) and in-band via [`TractBuffer`] status codes.

// The exported functions take opaque handles produced by this library; the
// caller is responsible for their validity, which makes the raw-pointer derefs
// expected here rather than a smell.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::rc::Rc;
use std::sync::Arc;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::model::TypedRunnableModel;

// tract only draws entropy for the unseeded RandomUniform/RandomNormal/Multinomial
// ops, none of which a classifier uses. Failing loudly beats returning fixed
// bytes that look random, and keeps tract_ffi.wasm free of any imports.
#[cfg(all(target_family = "wasm", any(target_os = "unknown", target_os = "none")))]
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

/// Buffer status: success.
pub const TRACT_STATUS_OK: i32 = 0;
/// Buffer status: model load or optimization failure.
pub const TRACT_STATUS_MODEL_ERROR: i32 = 1;
/// Buffer status: invalid arguments (null pointers, mismatched byte lengths, etc.).
pub const TRACT_STATUS_INVALID_INPUT: i32 = 2;
/// Buffer status: internal inference or panic error.
pub const TRACT_STATUS_INTERNAL_ERROR: i32 = 3;

/// Opaque result buffer handed across the portable C/WASM ABI as a raw pointer.
///
/// Rust allocates it, the caller reads it through the `tract_buffer_*`
/// accessors and must release it with [`tract_buffer_free`].
pub struct TractBuffer {
    data: Vec<u8>,
    status: i32,
}

// Datum type codes, mirrored by the Dart side (see `DatumType` in tract).
const DT_F32: i32 = 0;
const DT_F64: i32 = 1;
const DT_F16: i32 = 2;
const DT_I8: i32 = 3;
const DT_I16: i32 = 4;
const DT_I32: i32 = 5;
const DT_I64: i32 = 6;
const DT_U8: i32 = 7;
const DT_U16: i32 = 8;
const DT_U32: i32 = 9;
const DT_U64: i32 = 10;
const DT_BOOL: i32 = 11;

fn code_to_datum_type(code: i32) -> Option<DatumType> {
    Some(match code {
        DT_F32 => DatumType::F32,
        DT_F64 => DatumType::F64,
        DT_F16 => DatumType::F16,
        DT_I8 => DatumType::I8,
        DT_I16 => DatumType::I16,
        DT_I32 => DatumType::I32,
        DT_I64 => DatumType::I64,
        DT_U8 => DatumType::U8,
        DT_U16 => DatumType::U16,
        DT_U32 => DatumType::U32,
        DT_U64 => DatumType::U64,
        DT_BOOL => DatumType::Bool,
        _ => return None,
    })
}

fn datum_type_to_code(dt: DatumType) -> i32 {
    match dt {
        DatumType::F32 => DT_F32,
        DatumType::F64 => DT_F64,
        DatumType::F16 => DT_F16,
        DatumType::I8 => DT_I8,
        DatumType::I16 => DT_I16,
        DatumType::I32 => DT_I32,
        DatumType::I64 => DT_I64,
        DatumType::U8 => DT_U8,
        DatumType::U16 => DT_U16,
        DatumType::U32 => DT_U32,
        DatumType::U64 => DT_U64,
        DatumType::Bool => DT_BOOL,
        _ => -1,
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: impl Into<String>) {
    let c = CString::new(msg.into()).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Returns the most recent error message on the calling thread (never null;
/// an empty string when there is no error).
#[no_mangle]
pub extern "C" fn tract_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Returns the byte length (excluding the NUL terminator) of [`tract_last_error`].
#[no_mangle]
pub extern "C" fn tract_last_error_len() -> usize {
    LAST_ERROR.with(|e| e.borrow().as_bytes().len())
}

/// Runs `f`, turning a panic into an error. A panic must not unwind out of an
/// `extern "C"` fn - Rust aborts the whole process when one tries - so an
/// assertion tripped deep inside tract by a malformed model would otherwise
/// take the host app down with it.
fn catch_panic<T>(f: impl FnOnce() -> TractResult<T>) -> TractResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        Err(TractError::msg(format!("tract panicked: {msg}")))
    })
}

static VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns this library's version as a static NUL-terminated string.
///
/// The Dart `FfiBackend` also calls it to check that the library is present.
#[no_mangle]
pub extern "C" fn tract_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Pointer to the crate version as UTF-8 bytes (not NUL-terminated); the
/// length comes from [`tract_version_len`].
#[no_mangle]
pub extern "C" fn tract_version_ptr() -> *const u8 {
    VERSION.as_ptr()
}

/// Length of the [`tract_version_ptr`] bytes.
#[no_mangle]
pub extern "C" fn tract_version_len() -> usize {
    VERSION.len()
}

// ---------------------------------------------------------------------------
// Portable linear-memory staging & TractBuffer (native + wasm32)
// ---------------------------------------------------------------------------

/// Layout of a `tract_alloc(len)` block; `None` for
/// zero or oversized lengths. Aligned to 8 bytes so callers can stage `u8`,
/// `f32`, `f64`, `i64`, `usize` shapes, or pointer arrays in WASM linear memory.
fn staging_layout(len: usize) -> Option<std::alloc::Layout> {
    if len == 0 {
        return None;
    }
    std::alloc::Layout::from_size_align(len, 8).ok()
}

/// Allocates `len` bytes (8-byte aligned) and returns the pointer (null when
/// `len == 0` or on allocation failure). Callers stage input model bytes,
/// shapes, or tensor buffers here, then release with [`tract_free`] using the
/// same `len`.
#[no_mangle]
pub extern "C" fn tract_alloc(len: usize) -> *mut u8 {
    match staging_layout(len) {
        Some(layout) => unsafe { std::alloc::alloc(layout) },
        None => std::ptr::null_mut(),
    }
}

/// Releases memory obtained from [`tract_alloc`]. `len` must be the length
/// passed to `tract_alloc`; a null `ptr` is a no-op.
///
/// # Safety
/// `ptr` must come from `tract_alloc(len)` and not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn tract_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    if let Some(layout) = staging_layout(len) {
        std::alloc::dealloc(ptr, layout);
    }
}

/// Status code of a result buffer (`TRACT_STATUS_*`); `TRACT_STATUS_INTERNAL_ERROR`
/// for a null handle.
///
/// # Safety
/// `buf` must be null or a live handle returned by a `tract_infer_*` /
/// `tract_model_infer_*` call.
#[no_mangle]
pub unsafe extern "C" fn tract_buffer_status(buf: *const TractBuffer) -> i32 {
    match buf.as_ref() {
        Some(b) => b.status,
        None => TRACT_STATUS_INTERNAL_ERROR,
    }
}

/// Borrowed pointer to the buffer's payload bytes, valid until
/// [`tract_buffer_free`]. Null for a null handle.
///
/// On [`TRACT_STATUS_OK`], the payload is the contiguous native-endian bytes of
/// output tensor 0 (e.g. `f32` logits). On error, the payload is the UTF-8
/// error message.
///
/// # Safety
/// `buf` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn tract_buffer_data(buf: *const TractBuffer) -> *const u8 {
    match buf.as_ref() {
        Some(b) => b.data.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Length in bytes of the buffer's payload. 0 for a null handle.
///
/// # Safety
/// `buf` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn tract_buffer_len(buf: *const TractBuffer) -> usize {
    match buf.as_ref() {
        Some(b) => b.data.len(),
        None => 0,
    }
}

/// Frees a [`TractBuffer`]. Null-safe.
///
/// # Safety
/// `buf` must be null or a live handle; it must not be used or freed again
/// afterwards.
#[no_mangle]
pub unsafe extern "C" fn tract_buffer_free(buf: *mut TractBuffer) {
    if !buf.is_null() {
        drop(Box::from_raw(buf));
    }
}

fn ok_buffer(data: Vec<u8>) -> *mut TractBuffer {
    Box::into_raw(Box::new(TractBuffer {
        data,
        status: TRACT_STATUS_OK,
    }))
}

fn err_buffer(status: i32, msg: impl Into<String>) -> *mut TractBuffer {
    let s = msg.into();
    set_last_error(s.clone());
    Box::into_raw(Box::new(TractBuffer {
        data: s.into_bytes(),
        status,
    }))
}

// ---------------------------------------------------------------------------
// Tensors (values)
// ---------------------------------------------------------------------------

/// An owned tensor crossing the FFI boundary.
pub struct TractValue(Tensor);

/// Builds a tensor by copying `len` bytes of native-endian, row-major data with
/// the given datum type and shape. Returns null on error (see
/// [`tract_last_error`]); the returned handle must be freed with
/// [`tract_value_free`].
///
/// # Safety
/// `shape` must point to `rank` readable `usize`s (or be null when `rank` is 0),
/// and `data` must point to `len` readable bytes. `len` must equal
/// `product(shape) * sizeof(dtype)`.
#[no_mangle]
pub unsafe extern "C" fn tract_value_from_bytes(
    dtype: i32,
    shape: *const usize,
    rank: usize,
    data: *const c_void,
    len: usize,
) -> *mut TractValue {
    let Some(dt) = code_to_datum_type(dtype) else {
        set_last_error(format!("unknown datum type code {dtype}"));
        return std::ptr::null_mut();
    };
    let shape_slice: &[usize] = if rank == 0 {
        &[]
    } else if shape.is_null() {
        set_last_error("null shape with non-zero rank");
        return std::ptr::null_mut();
    } else {
        std::slice::from_raw_parts(shape, rank)
    };
    let Some(expected) = shape_slice
        .iter()
        .try_fold(dt.size_of(), |acc, &d| acc.checked_mul(d))
    else {
        set_last_error(format!("shape {shape_slice:?} overflows the address space"));
        return std::ptr::null_mut();
    };
    if len != expected {
        set_last_error(format!("data length {len} != expected {expected} bytes"));
        return std::ptr::null_mut();
    }
    let bytes: &[u8] = if len == 0 {
        &[]
    } else if data.is_null() {
        set_last_error("null data with non-zero length");
        return std::ptr::null_mut();
    } else {
        std::slice::from_raw_parts(data as *const u8, len)
    };
    // A Rust bool must be 0 or 1; anything else is undefined behavior once
    // tract reads the tensor.
    if dt == DatumType::Bool && bytes.iter().any(|&b| b > 1) {
        set_last_error("bool tensor data must be 0 or 1");
        return std::ptr::null_mut();
    }
    match catch_panic(|| Tensor::from_raw_dt(dt, shape_slice, bytes)) {
        Ok(t) => Box::into_raw(Box::new(TractValue(t))),
        Err(e) => {
            set_last_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Frees a tensor created by this library.
#[no_mangle]
pub extern "C" fn tract_value_free(value: *mut TractValue) {
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value) });
    }
}

fn value_ref<'a>(value: *const TractValue) -> Option<&'a TractValue> {
    if value.is_null() {
        set_last_error("null value");
        None
    } else {
        Some(unsafe { &*value })
    }
}

/// Returns a tensor's datum type code, or -1 if the handle is null.
#[no_mangle]
pub extern "C" fn tract_value_datum_type(value: *const TractValue) -> i32 {
    match value_ref(value) {
        Some(v) => datum_type_to_code(v.0.datum_type()),
        None => -1,
    }
}

/// Returns a tensor's rank (number of dimensions), or 0 if the handle is null.
#[no_mangle]
pub extern "C" fn tract_value_rank(value: *const TractValue) -> usize {
    value_ref(value).map(|v| v.0.rank()).unwrap_or(0)
}

/// Writes up to `cap` dimensions of a tensor's shape into `out` and returns the
/// tensor's full rank (which may exceed `cap`). Returns 0 if the handle is null.
///
/// # Safety
/// `out` must point to `cap` writable `usize`s (or be null when `cap` is 0).
#[no_mangle]
pub unsafe extern "C" fn tract_value_shape(
    value: *const TractValue,
    out: *mut usize,
    cap: usize,
) -> usize {
    let Some(v) = value_ref(value) else {
        return 0;
    };
    let shape = v.0.shape();
    if !out.is_null() && cap > 0 {
        let n = shape.len().min(cap);
        std::ptr::copy_nonoverlapping(shape.as_ptr(), out, n);
    }
    shape.len()
}

/// Returns a tensor's element count, or 0 if the handle is null.
#[no_mangle]
pub extern "C" fn tract_value_len(value: *const TractValue) -> usize {
    value_ref(value).map(|v| v.0.len()).unwrap_or(0)
}

/// Returns the length in bytes of a tensor's data buffer, or 0 if null.
#[no_mangle]
pub extern "C" fn tract_value_byte_len(value: *const TractValue) -> usize {
    value_ref(value).map(|v| v.0.as_bytes().len()).unwrap_or(0)
}

/// Returns a borrowed pointer to a tensor's contiguous, native-endian data,
/// valid until the tensor is freed. Returns null if the handle is null.
#[no_mangle]
pub extern "C" fn tract_value_data(value: *const TractValue) -> *const c_void {
    match value_ref(value) {
        Some(v) => v.0.as_bytes().as_ptr() as *const c_void,
        None => std::ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A loaded, optimized, runnable model and its input/output names.
pub struct TractModel {
    plan: Arc<TypedRunnableModel>,
    input_names: Vec<CString>,
    output_names: Vec<CString>,
    norm: Normalization,
}

/// Per-model input normalization for the pixel entrypoints
/// ([`tract_model_run_u8`], [`tract_model_run_f32`], [`tract_infer_28x28`]):
/// `v = if invert { max - v } else { v }; out = (v * scale - mean) / std`.
/// The default is the identity.
#[derive(Clone, Copy)]
struct Normalization {
    scale: f32,
    mean: f32,
    std: f32,
    max: f32,
    invert: bool,
}

impl Normalization {
    const IDENTITY: Normalization = Normalization {
        scale: 1.0,
        mean: 0.0,
        std: 1.0,
        max: 255.0,
        invert: false,
    };

    fn apply(&self, v: f32) -> f32 {
        let v = if self.invert { self.max - v } else { v };
        (v * self.scale - self.mean) / self.std
    }
}

fn names_for(model: &TypedModel, outlets: &[OutletId]) -> Vec<CString> {
    outlets
        .iter()
        .map(|o| CString::new(model.node(o.node).name.as_str()).unwrap_or_default())
        .collect()
}

fn make_model(plan: Arc<TypedRunnableModel>) -> TractResult<TractModel> {
    let model = plan.model();
    let input_names = names_for(model, model.input_outlets()?);
    let output_names = names_for(model, model.output_outlets()?);
    Ok(TractModel {
        plan,
        input_names,
        output_names,
        norm: Normalization::IDENTITY,
    })
}

fn finalize_inference(model: InferenceModel) -> TractResult<TractModel> {
    make_model(model.into_optimized()?.into_runnable()?)
}

fn finalize_typed(model: TypedModel) -> TractResult<TractModel> {
    make_model(model.into_optimized()?.into_runnable()?)
}

fn to_model(build: impl FnOnce() -> TractResult<TractModel>) -> *mut TractModel {
    match catch_panic(build) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_last_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

unsafe fn read_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        set_last_error("null string");
        return None;
    }
    match CStr::from_ptr(ptr).to_str() {
        Ok(s) => Some(s),
        Err(_) => {
            set_last_error("string is not valid UTF-8");
            None
        }
    }
}

/// Loads and optimizes an ONNX model from a file path. Returns null on error;
/// free with [`tract_model_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_load_path(path: *const c_char) -> *mut TractModel {
    let Some(path) = read_str(path) else {
        return std::ptr::null_mut();
    };
    to_model(|| {
        tract_onnx::onnx()
            .model_for_path(path)
            .and_then(finalize_inference)
    })
}

/// Like [`tract_onnx_model_load_path`] but pins input 0 to `dtype` and the given
/// `shape` first, resolving symbolic dimensions (e.g. a dynamic batch size) so
/// the model can be optimized.
///
/// # Safety
/// `path` must be a valid NUL-terminated C string and `shape` must point to
/// `rank` readable `usize`s (or be null when `rank` is 0).
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_load_path_with_shape(
    path: *const c_char,
    dtype: i32,
    shape: *const usize,
    rank: usize,
) -> *mut TractModel {
    let Some(path) = read_str(path) else {
        return std::ptr::null_mut();
    };
    let Some(dt) = code_to_datum_type(dtype) else {
        set_last_error(format!("unknown datum type code {dtype}"));
        return std::ptr::null_mut();
    };
    let shape_vec: Vec<usize> = if rank == 0 {
        Vec::new()
    } else if shape.is_null() {
        set_last_error("null shape with non-zero rank");
        return std::ptr::null_mut();
    } else {
        std::slice::from_raw_parts(shape, rank).to_vec()
    };
    let build = || -> TractResult<TractModel> {
        let model = tract_onnx::onnx().model_for_path(path)?;
        let model = model.with_input_fact(0, InferenceFact::dt_shape(dt, shape_vec))?;
        finalize_inference(model)
    };
    to_model(build)
}

/// Loads and optimizes an ONNX model from an in-memory buffer. Returns null on
/// error; free with [`tract_model_free`].
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_load_bytes(
    data: *const c_void,
    len: usize,
) -> *mut TractModel {
    if data.is_null() {
        set_last_error("null data");
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, len);
    let mut reader = std::io::Cursor::new(bytes);
    to_model(|| {
        tract_onnx::onnx()
            .model_for_read(&mut reader)
            .and_then(finalize_inference)
    })
}

/// Like [`tract_onnx_model_load_bytes`] but pins input 0 to `dtype` and the
/// given `shape` first, resolving symbolic dimensions (e.g. a dynamic batch
/// size) so the model can be optimized.
///
/// # Safety
/// `data` must point to `len` readable bytes and `shape` to `rank` readable
/// `usize`s (or be null when `rank` is 0).
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_load_bytes_with_shape(
    data: *const c_void,
    len: usize,
    dtype: i32,
    shape: *const usize,
    rank: usize,
) -> *mut TractModel {
    if data.is_null() {
        set_last_error("null data");
        return std::ptr::null_mut();
    }
    let Some(dt) = code_to_datum_type(dtype) else {
        set_last_error(format!("unknown datum type code {dtype}"));
        return std::ptr::null_mut();
    };
    let shape_vec: Vec<usize> = if rank == 0 {
        Vec::new()
    } else if shape.is_null() {
        set_last_error("null shape with non-zero rank");
        return std::ptr::null_mut();
    } else {
        std::slice::from_raw_parts(shape, rank).to_vec()
    };
    let bytes = std::slice::from_raw_parts(data as *const u8, len);
    let build = || -> TractResult<TractModel> {
        let mut reader = std::io::Cursor::new(bytes);
        let model = tract_onnx::onnx().model_for_read(&mut reader)?;
        let model = model.with_input_fact(0, InferenceFact::dt_shape(dt, shape_vec))?;
        finalize_inference(model)
    };
    to_model(build)
}

/// Loads and optimizes a tract NNEF model from a file path (a directory or a
/// `.nnef.tgz` archive). Returns null on error; free with [`tract_model_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_model_load_path(path: *const c_char) -> *mut TractModel {
    let Some(path) = read_str(path) else {
        return std::ptr::null_mut();
    };
    to_model(|| {
        tract_nnef::nnef()
            .model_for_path(path)
            .and_then(finalize_typed)
    })
}

/// Loads and optimizes a tract NNEF model from an in-memory `.nnef.tgz` archive.
/// Returns null on error; free with [`tract_model_free`].
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_model_load_bytes(
    data: *const c_void,
    len: usize,
) -> *mut TractModel {
    if data.is_null() {
        set_last_error("null data");
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, len);
    let mut reader = std::io::Cursor::new(bytes);
    to_model(|| {
        tract_nnef::nnef()
            .model_for_read(&mut reader)
            .and_then(finalize_typed)
    })
}

/// Frees a model created by this library.
#[no_mangle]
pub extern "C" fn tract_model_free(model: *mut TractModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

fn model_ref<'a>(model: *const TractModel) -> Option<&'a TractModel> {
    if model.is_null() {
        set_last_error("null model");
        None
    } else {
        Some(unsafe { &*model })
    }
}

/// Returns the number of inputs the model expects, or 0 if the handle is null.
#[no_mangle]
pub extern "C" fn tract_model_input_count(model: *const TractModel) -> usize {
    model_ref(model).map(|m| m.input_names.len()).unwrap_or(0)
}

/// Returns the number of outputs the model produces, or 0 if the handle is null.
#[no_mangle]
pub extern "C" fn tract_model_output_count(model: *const TractModel) -> usize {
    model_ref(model).map(|m| m.output_names.len()).unwrap_or(0)
}

/// Returns the name of input `idx` as a borrowed NUL-terminated string valid
/// until the model is freed, or null if the handle/index is invalid.
#[no_mangle]
pub extern "C" fn tract_model_input_name(model: *const TractModel, idx: usize) -> *const c_char {
    match model_ref(model).and_then(|m| m.input_names.get(idx)) {
        Some(name) => name.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Returns the name of output `idx` as a borrowed NUL-terminated string valid
/// until the model is freed, or null if the handle/index is invalid.
#[no_mangle]
pub extern "C" fn tract_model_output_name(model: *const TractModel, idx: usize) -> *const c_char {
    match model_ref(model).and_then(|m| m.output_names.get(idx)) {
        Some(name) => name.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Runs inference. Reads `n_inputs` input tensor handles from `inputs` (which
/// must match [`tract_model_input_count`]), writes the resulting output handles
/// into `outputs` (capacity `out_cap`), and returns the number of outputs
/// written, or -1 on error (see [`tract_last_error`]).
///
/// The caller owns the returned output tensors and must free each with
/// [`tract_value_free`]. The input tensors are not consumed.
///
/// # Safety
/// `inputs` must point to `n_inputs` readable, non-null [`TractValue`] handles
/// and `outputs` to `out_cap` writable pointers.
#[no_mangle]
pub unsafe extern "C" fn tract_model_run(
    model: *mut TractModel,
    inputs: *const *const TractValue,
    n_inputs: usize,
    outputs: *mut *mut TractValue,
    out_cap: usize,
) -> isize {
    let Some(model) = model_ref(model) else {
        return -1;
    };
    if inputs.is_null() && n_inputs != 0 {
        set_last_error("null inputs");
        return -1;
    }
    if outputs.is_null() && out_cap != 0 {
        set_last_error("null outputs");
        return -1;
    }
    let in_handles = if n_inputs == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(inputs, n_inputs)
    };
    let mut values: TVec<TValue> = tvec!();
    for &handle in in_handles {
        let Some(v) = value_ref(handle) else {
            return -1;
        };
        values.push(v.0.clone().into());
    }
    match catch_panic(|| model.plan.run(values)) {
        Ok(result) => {
            let n = result.len();
            if n > out_cap {
                set_last_error(format!(
                    "output buffer too small: {n} outputs, capacity {out_cap}"
                ));
                return -1;
            }
            // Fail before handing out any handles, so the caller never has to
            // free a partial set.
            if let Some((ix, dt)) = result
                .iter()
                .map(|t| t.datum_type())
                .enumerate()
                .find(|&(_, dt)| datum_type_to_code(dt) < 0)
            {
                set_last_error(format!("output {ix} has unsupported datum type {dt:?}"));
                return -1;
            }
            // n <= out_cap, so a null `outputs` (only allowed with out_cap 0)
            // is never written through.
            for (ix, tv) in result.into_iter().enumerate() {
                *outputs.add(ix) = Box::into_raw(Box::new(TractValue(tv.into_tensor())));
            }
            n as isize
        }
        Err(e) => {
            set_last_error(e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Model handles & buffer-based inference (native + wasm32)
// ---------------------------------------------------------------------------

/// Load option flag: invert pixels (`max - v`) before scaling.
pub const TRACT_LOAD_INVERT: u32 = 1;

/// Loads an ONNX model from `data[..len]` once and returns a handle for
/// [`tract_model_run_f32`], [`tract_model_run_u8`] and [`tract_infer_28x28`],
/// so the model is parsed and optimized a single time. Returns null on error
/// (message via [`tract_last_error`]); release with [`tract_model_free`].
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tract_model_load(data: *const u8, len: usize) -> *mut TractModel {
    tract_model_load_with_options(data, len, 0, 1.0, 0.0, 1.0, 255.0, std::ptr::null(), 0)
}

/// Like [`tract_model_load`] with input options.
///
/// - `flags`: [`TRACT_LOAD_INVERT`] inverts each pixel (`max - v`) first.
/// - `scale`, `mean`, `std`, `max`: every input value `v` the pixel
///   entrypoints see (raw `u8`, or `f32` in the same raw units) becomes
///   `((v or max - v) * scale - mean) / std`. `std` must be finite and
///   non-zero. Identity is `scale 1, mean 0, std 1`; `max` is `255` for 8-bit
///   pixels.
/// - `shape`/`rank`: when `rank > 0`, pins input 0 to `f32` with this shape
///   (needed for models with symbolic dimensions). Null with `rank == 0`
///   keeps the model's declared shape.
///
/// For the common MNIST convention `(x / 255 - 0.1307) / 0.3081` pass
/// `scale = 1/255, mean = 0.1307, std = 0.3081`. The bundled `mnist-8.onnx`
/// itself works with plain `x / 255` and with the MNIST statistics, but not
/// well with raw 0..255.
///
/// # Safety
/// `data` must point to `len` readable bytes and `shape` to `rank` readable
/// `usize`s (or be null when `rank` is 0).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tract_model_load_with_options(
    data: *const u8,
    len: usize,
    flags: u32,
    scale: f32,
    mean: f32,
    std: f32,
    max: f32,
    shape: *const usize,
    rank: usize,
) -> *mut TractModel {
    if data.is_null() || len == 0 {
        set_last_error("empty or null model bytes");
        return std::ptr::null_mut();
    }
    if !(std.is_finite() && std != 0.0 && scale.is_finite() && mean.is_finite() && max.is_finite())
    {
        set_last_error("normalization values must be finite and std non-zero");
        return std::ptr::null_mut();
    }
    let shape_vec = if rank == 0 {
        Vec::new()
    } else if shape.is_null() {
        set_last_error("null shape with non-zero rank");
        return std::ptr::null_mut();
    } else {
        std::slice::from_raw_parts(shape, rank).to_vec()
    };
    let bytes = std::slice::from_raw_parts(data, len);
    let norm = Normalization {
        scale,
        mean,
        std,
        max,
        invert: flags & TRACT_LOAD_INVERT != 0,
    };
    to_model(|| load_onnx_pinned(bytes, shape_vec).map(|m| TractModel { norm, ..m }))
}

fn load_onnx_pinned(bytes: &[u8], shape: Vec<usize>) -> TractResult<TractModel> {
    let mut reader = std::io::Cursor::new(bytes);
    let mut model = tract_onnx::onnx().model_for_read(&mut reader)?;
    if !shape.is_empty() {
        model = model.with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, shape))?;
    }
    finalize_inference(model)
}

fn run_plan_to_buffer(plan: &Arc<TypedRunnableModel>, input: Tensor) -> *mut TractBuffer {
    match catch_panic(|| plan.run(tvec!(input.into()))) {
        Ok(mut outputs) => {
            if outputs.is_empty() {
                return err_buffer(TRACT_STATUS_INTERNAL_ERROR, "model produced 0 outputs");
            }
            let first = outputs.remove(0).into_tensor();
            // If output 0 is convertible to f32 (or already f32), emit contiguous
            // native-endian f32 bytes; otherwise emit its raw tensor bytes.
            match first.cast_to_dt(DatumType::F32) {
                Ok(casted) => ok_buffer(casted.as_bytes().to_vec()),
                Err(_) => ok_buffer(first.as_bytes().to_vec()),
            }
        }
        Err(e) => err_buffer(TRACT_STATUS_INTERNAL_ERROR, e.to_string()),
    }
}

/// Reads a caller-provided shape; an error buffer for a null/overflowing one.
unsafe fn read_shape<'a>(
    shape: *const usize,
    rank: usize,
) -> Result<(&'a [usize], usize), *mut TractBuffer> {
    let dims: &[usize] = if rank == 0 {
        &[]
    } else if shape.is_null() {
        return Err(err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            "null shape with non-zero rank",
        ));
    } else {
        std::slice::from_raw_parts(shape, rank)
    };
    match dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)) {
        Some(n) => Ok((dims, n)),
        None => Err(err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            format!("shape {dims:?} overflows the address space"),
        )),
    }
}

/// Runs `model` on `count` already-collected values after normalizing them.
fn run_values(
    model: &TractModel,
    shape: &[usize],
    values: impl Iterator<Item = f32>,
) -> *mut TractBuffer {
    let norm = model.norm;
    let values: Vec<f32> = values.map(|v| norm.apply(v)).collect();
    match catch_panic(|| Tensor::from_shape(shape, &values)) {
        Ok(t) => run_plan_to_buffer(&model.plan, t),
        Err(e) => err_buffer(TRACT_STATUS_INVALID_INPUT, e.to_string()),
    }
}

fn run_f32_bytes(model: &TractModel, shape: &[usize], bytes: &[u8]) -> *mut TractBuffer {
    run_values(
        model,
        shape,
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])),
    )
}

fn run_u8_bytes(model: &TractModel, shape: &[usize], bytes: &[u8]) -> *mut TractBuffer {
    run_values(model, shape, bytes.iter().map(|&b| b as f32))
}

/// Shared checks for the `tract_model_run_*` entrypoints, then `run`.
unsafe fn run_checked(
    model: *const TractModel,
    data: *const u8,
    data_len: usize,
    elem_size: usize,
    shape: *const usize,
    rank: usize,
    run: fn(&TractModel, &[usize], &[u8]) -> *mut TractBuffer,
) -> *mut TractBuffer {
    let Some(model) = model_ref(model) else {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "null model");
    };
    let (dims, count) = match read_shape(shape, rank) {
        Ok(v) => v,
        Err(buf) => return buf,
    };
    let Some(expected) = count.checked_mul(elem_size) else {
        return err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            format!("shape {dims:?} overflows the address space"),
        );
    };
    if data_len != expected {
        return err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            format!("data length {data_len} != expected {expected} bytes"),
        );
    }
    if data_len != 0 && data.is_null() {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "null data with non-zero length");
    }
    let bytes: &[u8] = if data_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data, data_len)
    };
    run(model, dims, bytes)
}

/// Runs a model handle on a single `f32` input of `shape[..rank]` given as
/// `data[..data_len]` native-endian bytes (`data_len` must be
/// `product(shape) * 4`), applying the handle's input normalization, and
/// returns output 0's `f32` bytes in a [`TractBuffer`]. Never returns null;
/// check [`tract_buffer_status`] and release with [`tract_buffer_free`].
///
/// # Safety
/// `model` must be a live handle (null yields an error buffer), `data` must
/// point to `data_len` readable bytes and `shape` to `rank` readable `usize`s.
#[no_mangle]
pub unsafe extern "C" fn tract_model_run_f32(
    model: *const TractModel,
    data: *const u8,
    data_len: usize,
    shape: *const usize,
    rank: usize,
) -> *mut TractBuffer {
    run_checked(model, data, data_len, 4, shape, rank, run_f32_bytes)
}

/// Like [`tract_model_run_f32`] but the input is `u8` pixels (`data_len` must be
/// `product(shape)`), the usual case for canvas/stylus images.
///
/// # Safety
/// Same contract as [`tract_model_run_f32`].
#[no_mangle]
pub unsafe extern "C" fn tract_model_run_u8(
    model: *const TractModel,
    data: *const u8,
    data_len: usize,
    shape: *const usize,
    rank: usize,
) -> *mut TractBuffer {
    run_checked(model, data, data_len, 1, shape, rank, run_u8_bytes)
}

/// Same as [`tract_model_run_f32`] with the arguments in the older order.
///
/// # Safety
/// Same contract as [`tract_model_run_f32`].
#[no_mangle]
pub unsafe extern "C" fn tract_model_infer_f32(
    model: *mut TractModel,
    shape: *const usize,
    rank: usize,
    data: *const u8,
    data_len: usize,
) -> *mut TractBuffer {
    tract_model_run_f32(model, data, data_len, shape, rank)
}

const PIXELS_28X28: usize = 28 * 28;

fn shape_for_28x28_typed(model: &TypedModel) -> Vec<usize> {
    if let Ok(fact) = model.input_fact(0) {
        if let Some(dims) = fact.shape.as_concrete() {
            if dims.iter().copied().product::<usize>() == PIXELS_28X28 {
                return dims.to_vec();
            }
        }
    }
    vec![1, 1, 28, 28]
}

fn run_28x28(model: &TractModel, shape: &[usize], pixels: &[u8]) -> *mut TractBuffer {
    match pixels.len() {
        PIXELS_28X28 => run_u8_bytes(model, shape, pixels),
        n if n == PIXELS_28X28 * 4 => run_f32_bytes(model, shape, pixels),
        n => err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            format!("expected 784 (u8) or 3136 (f32) bytes for a 28x28 input, got {n}"),
        ),
    }
}

/// Runs a loaded [`TractModel`] on a 28x28 handwritten digit/glyph image
/// (`784` `u8` grayscale bytes `0..=255` or `3136` native-endian `f32` bytes),
/// applying the handle's input normalization (none unless set through
/// [`tract_model_load_with_options`]) and returning output 0's `f32` logits in
/// a [`TractBuffer`].
///
/// # Safety
/// `model` must be a valid [`TractModel`] handle (or null, which returns an
/// error buffer), and `pixels` must point to `pixels_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tract_infer_28x28(
    model: *mut TractModel,
    pixels: *const u8,
    pixels_len: usize,
) -> *mut TractBuffer {
    let Some(model) = model_ref(model) else {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "null model");
    };
    if pixels.is_null() {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "null pixels pointer");
    }
    let shape = shape_for_28x28_typed(model.plan.model());
    run_28x28(
        model,
        &shape,
        std::slice::from_raw_parts(pixels, pixels_len),
    )
}

/// Exact cache key: the full model bytes (compared, not hashed, so a lookup
/// can never return a different model) and the pinned input shape.
#[derive(PartialEq, Eq)]
struct CachedModelKey {
    model: Vec<u8>,
    input_shape: Vec<usize>,
}

thread_local! {
    static CACHED_MODEL: RefCell<Option<(CachedModelKey, Rc<TractModel>)>> =
        const { RefCell::new(None) };
}

/// One-shot inference: loads (or reuses from a one-entry thread-local cache)
/// the ONNX model in `model_bytes[..model_len]` and runs
/// `input_bytes[..input_len]` through it, returning output 0's `f32` bytes in
/// a [`TractBuffer`]. No input normalization is applied.
///
/// The cache keeps a copy of the model bytes and compares them in full on every
/// call (exact, but O(model size) per call); prefer [`tract_model_load`] and
/// [`tract_model_run_f32`] when running a model repeatedly.
///
/// Input conventions:
/// - `input_len == 784` (784 `u8` pixels `0..=255`) or `input_len == 3136`
///   (784 `f32` pixels): runs with a 28x28 image tensor (`[1, 1, 28, 28]`).
/// - Any other non-zero multiple of 4 (`N = input_len / 4` `f32` values):
///   runs with a 1-D `[N]` `f32` tensor (pinning input 0 to `[N]`).
///
/// # Safety
/// `model_bytes` must point to `model_len` readable bytes and `input_bytes` to
/// `input_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tract_infer_bytes(
    model_bytes: *const u8,
    model_len: usize,
    input_bytes: *const u8,
    input_len: usize,
) -> *mut TractBuffer {
    if model_bytes.is_null() || model_len == 0 {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "empty or null model bytes");
    }
    if input_bytes.is_null() || input_len == 0 {
        return err_buffer(TRACT_STATUS_INVALID_INPUT, "empty or null input bytes");
    }
    let onnx = std::slice::from_raw_parts(model_bytes, model_len);
    let input = std::slice::from_raw_parts(input_bytes, input_len);
    let is_28x28 = input_len == PIXELS_28X28 || input_len == PIXELS_28X28 * 4;
    if !is_28x28 && !input_len.is_multiple_of(4) {
        return err_buffer(
            TRACT_STATUS_INVALID_INPUT,
            format!(
                "input length {input_len} must be 784 (u8 28x28) or a multiple of 4 (f32 bytes)"
            ),
        );
    }

    let target_shape: Vec<usize> = if is_28x28 {
        vec![1, 1, 28, 28]
    } else {
        vec![input_len / 4]
    };

    let cached = CACHED_MODEL.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|(k, _)| k.input_shape == target_shape && k.model == onnx)
            .map(|(_, m)| Rc::clone(m))
    });

    let model = match cached {
        Some(model) => model,
        None => match catch_panic(|| load_onnx_pinned(onnx, target_shape.clone())) {
            Ok(m) => {
                let model = Rc::new(m);
                let key = CachedModelKey {
                    model: onnx.to_vec(),
                    input_shape: target_shape.clone(),
                };
                CACHED_MODEL.with(|cell| *cell.borrow_mut() = Some((key, Rc::clone(&model))));
                model
            }
            Err(e) => return err_buffer(TRACT_STATUS_MODEL_ERROR, e.to_string()),
        },
    };

    if is_28x28 {
        run_28x28(&model, &target_shape, input)
    } else {
        run_f32_bytes(&model, &target_shape, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the C ABI end to end on a trivial identity model (output = the
    // single source), without needing an ONNX/NNEF file on disk.
    #[test]
    fn identity_roundtrip_via_c_abi() {
        let mut typed = TypedModel::default();
        let sample = Tensor::zero_dt(DatumType::F32, &[2, 2]).unwrap();
        let fact = TypedFact::shape_and_dt_of(&sample);
        let src = typed.add_source("x", fact).unwrap();
        typed.select_output_outlets(&[src]).unwrap();
        let model = Box::into_raw(Box::new(finalize_typed(typed).unwrap()));

        assert_eq!(tract_model_input_count(model), 1);
        assert_eq!(tract_model_output_count(model), 1);

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let shape: [usize; 2] = [2, 2];
        let input = unsafe {
            tract_value_from_bytes(
                DT_F32,
                shape.as_ptr(),
                2,
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(&data),
            )
        };
        assert!(!input.is_null());

        let inputs: [*const TractValue; 1] = [input];
        let mut outputs: [*mut TractValue; 1] = [std::ptr::null_mut()];
        let n = unsafe { tract_model_run(model, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1) };
        assert_eq!(n, 1);

        let out = outputs[0];
        assert!(!out.is_null());
        assert_eq!(tract_value_datum_type(out), DT_F32);
        assert_eq!(tract_value_rank(out), 2);
        assert_eq!(tract_value_len(out), 4);

        let ptr = tract_value_data(out) as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, 4) };
        assert_eq!(slice, &data);

        tract_value_free(input);
        tract_value_free(out);
        tract_model_free(model);
    }

    #[test]
    fn value_from_bytes_rejects_overflowing_shape() {
        // The byte length wraps to a small number in release builds if the
        // multiplication is unchecked.
        let shape: [usize; 2] = [usize::MAX, usize::MAX];
        let data = [0u8; 4];
        let v = unsafe {
            tract_value_from_bytes(DT_F32, shape.as_ptr(), 2, data.as_ptr() as *const c_void, 4)
        };
        assert!(v.is_null());
        let err = unsafe { CStr::from_ptr(tract_last_error()) };
        assert!(err.to_str().unwrap().contains("overflows"));
    }

    #[test]
    fn value_from_bytes_rejects_invalid_bool() {
        let shape: [usize; 1] = [3];
        let bad = [0u8, 1, 2];
        let v = unsafe {
            tract_value_from_bytes(DT_BOOL, shape.as_ptr(), 1, bad.as_ptr() as *const c_void, 3)
        };
        assert!(v.is_null());

        let good = [0u8, 1, 1];
        let v = unsafe {
            tract_value_from_bytes(
                DT_BOOL,
                shape.as_ptr(),
                1,
                good.as_ptr() as *const c_void,
                3,
            )
        };
        assert!(!v.is_null());
        tract_value_free(v);
    }

    #[test]
    fn onnx_load_with_shape_pins_input_zero() {
        // A float32[3] -> float32[3] Identity model, shared with the Dart tests.
        // Pinning input 0 to [4] overrides the declared shape, so the model then
        // accepts a 4-element input.
        let onnx = include_bytes!("../../test/fixtures/identity.onnx");
        let shape: [usize; 1] = [4];
        let model = unsafe {
            tract_onnx_model_load_bytes_with_shape(
                onnx.as_ptr() as *const c_void,
                onnx.len(),
                DT_F32,
                shape.as_ptr(),
                1,
            )
        };
        assert!(!model.is_null());

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input = unsafe {
            tract_value_from_bytes(
                DT_F32,
                shape.as_ptr(),
                1,
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(&data),
            )
        };
        let inputs: [*const TractValue; 1] = [input];
        let mut outputs: [*mut TractValue; 1] = [std::ptr::null_mut()];
        let n = unsafe { tract_model_run(model, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1) };
        assert_eq!(n, 1);
        assert_eq!(tract_value_len(outputs[0]), 4);

        tract_value_free(input);
        tract_value_free(outputs[0]);
        tract_model_free(model);
    }

    #[test]
    fn catch_panic_reports_instead_of_unwinding() {
        let err = catch_panic::<()>(|| panic!("boom")).unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn loading_garbage_onnx_fails_cleanly() {
        let garbage = b"not a real onnx model";
        let model = unsafe {
            tract_onnx_model_load_bytes(garbage.as_ptr() as *const c_void, garbage.len())
        };
        assert!(model.is_null());
        let err = unsafe { CStr::from_ptr(tract_last_error()) };
        assert!(!err.to_bytes().is_empty());
        assert_eq!(tract_last_error_len(), err.to_bytes().len());
    }

    #[test]
    fn wasm_alloc_and_version_span_work() {
        let v_bytes =
            unsafe { std::slice::from_raw_parts(tract_version_ptr(), tract_version_len()) };
        assert_eq!(
            std::str::from_utf8(v_bytes).unwrap(),
            env!("CARGO_PKG_VERSION")
        );

        assert!(tract_alloc(0).is_null());
        unsafe {
            tract_free(std::ptr::null_mut(), 0);
        }

        let ptr = tract_alloc(37);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0, "staging buffer must be 8-byte aligned");
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 37);
            assert_eq!(*ptr.add(36), 0xAB);
            tract_free(ptr, 37);
        }
    }

    #[test]
    fn buffer_null_handles_and_errors_are_safe() {
        unsafe {
            assert_eq!(
                tract_buffer_status(std::ptr::null()),
                TRACT_STATUS_INTERNAL_ERROR
            );
            assert!(tract_buffer_data(std::ptr::null()).is_null());
            assert_eq!(tract_buffer_len(std::ptr::null()), 0);
            tract_buffer_free(std::ptr::null_mut());

            let bad = b"not onnx";
            let input = [1.0f32, 2.0, 3.0];
            let buf = tract_infer_bytes(
                bad.as_ptr(),
                bad.len(),
                input.as_ptr() as *const u8,
                std::mem::size_of_val(&input),
            );
            assert_eq!(tract_buffer_status(buf), TRACT_STATUS_MODEL_ERROR);
            let msg_bytes =
                std::slice::from_raw_parts(tract_buffer_data(buf), tract_buffer_len(buf));
            assert!(!msg_bytes.is_empty());
            tract_buffer_free(buf);
        }
    }

    #[test]
    fn infer_bytes_and_model_infer_f32_roundtrip_identity() {
        let onnx = include_bytes!("../../test/fixtures/identity.onnx");
        let input: [f32; 3] = [1.25, -2.5, 4.75];
        let input_bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, std::mem::size_of_val(&input))
        };

        // 1. One-shot / cached tract_infer_bytes via staged WASM-style buffers
        unsafe {
            let m_ptr = tract_alloc(onnx.len());
            std::ptr::copy_nonoverlapping(onnx.as_ptr(), m_ptr, onnx.len());
            let i_ptr = tract_alloc(input_bytes.len());
            std::ptr::copy_nonoverlapping(input_bytes.as_ptr(), i_ptr, input_bytes.len());

            let buf = tract_infer_bytes(m_ptr, onnx.len(), i_ptr, input_bytes.len());
            tract_free(m_ptr, onnx.len());
            tract_free(i_ptr, input_bytes.len());

            assert_eq!(tract_buffer_status(buf), TRACT_STATUS_OK);
            assert_eq!(tract_buffer_len(buf), std::mem::size_of_val(&input));
            let out_f32 =
                std::slice::from_raw_parts(tract_buffer_data(buf) as *const f32, input.len());
            assert_eq!(out_f32, &input);
            tract_buffer_free(buf);
        }

        // 2. Stateful tract_model_infer_f32
        unsafe {
            let model = tract_onnx_model_load_bytes(onnx.as_ptr() as *const c_void, onnx.len());
            assert!(!model.is_null());
            let shape: [usize; 1] = [3];
            let buf = tract_model_infer_f32(
                model,
                shape.as_ptr(),
                1,
                input_bytes.as_ptr(),
                input_bytes.len(),
            );
            assert_eq!(tract_buffer_status(buf), TRACT_STATUS_OK);
            let out_f32 =
                std::slice::from_raw_parts(tract_buffer_data(buf) as *const f32, input.len());
            assert_eq!(out_f32, &input);
            tract_buffer_free(buf);
            tract_model_free(model);
        }
    }

    #[test]
    fn infer_28x28_and_infer_bytes_classify_real_mnist_digits() {
        let onnx = include_bytes!("../../example/assets/mnist-8.onnx");
        let samples = include_bytes!("../../example/assets/mnist_samples.bin");
        const STRIDE: usize = 1 + 28 * 28;
        let sample_count = samples.len() / STRIDE;
        assert_eq!(sample_count, 10);

        let model =
            unsafe { tract_onnx_model_load_bytes(onnx.as_ptr() as *const c_void, onnx.len()) };
        assert!(!model.is_null());

        for idx in 0..sample_count {
            let offset = idx * STRIDE;
            let expected_label = samples[offset] as usize;
            let pixels_u8 = &samples[offset + 1..offset + STRIDE];

            // Test A: tract_infer_28x28 with 784 u8 grayscale bytes
            unsafe {
                let buf = tract_infer_28x28(model, pixels_u8.as_ptr(), pixels_u8.len());
                assert_eq!(tract_buffer_status(buf), TRACT_STATUS_OK);
                assert_eq!(tract_buffer_len(buf), 10 * std::mem::size_of::<f32>());
                let logits = std::slice::from_raw_parts(tract_buffer_data(buf) as *const f32, 10);
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                assert_eq!(pred, expected_label, "u8 28x28 mismatch at sample {idx}");
                tract_buffer_free(buf);
            }

            // Test B: tract_infer_bytes with 3136 f32 bytes (and cached ONNX plan)
            let pixels_f32: Vec<f32> = pixels_u8.iter().map(|&b| b as f32).collect();
            unsafe {
                let buf = tract_infer_bytes(
                    onnx.as_ptr(),
                    onnx.len(),
                    pixels_f32.as_ptr() as *const u8,
                    pixels_f32.len() * std::mem::size_of::<f32>(),
                );
                assert_eq!(tract_buffer_status(buf), TRACT_STATUS_OK);
                assert_eq!(tract_buffer_len(buf), 10 * std::mem::size_of::<f32>());
                let logits = std::slice::from_raw_parts(tract_buffer_data(buf) as *const f32, 10);
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                assert_eq!(pred, expected_label, "f32 28x28 mismatch at sample {idx}");
                tract_buffer_free(buf);
            }
        }

        tract_model_free(model);
    }

    unsafe fn logits(buf: *mut TractBuffer) -> Vec<f32> {
        assert_eq!(tract_buffer_status(buf), TRACT_STATUS_OK);
        let n = tract_buffer_len(buf) / 4;
        let v = std::slice::from_raw_parts(tract_buffer_data(buf) as *const f32, n).to_vec();
        tract_buffer_free(buf);
        v
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    #[test]
    fn model_handle_runs_u8_and_f32_with_normalization() {
        let onnx = include_bytes!("../../example/assets/mnist-8.onnx");
        let samples = include_bytes!("../../example/assets/mnist_samples.bin");
        const STRIDE: usize = 1 + 28 * 28;
        let shape: [usize; 4] = [1, 1, 28, 28];
        unsafe {
            let plain = tract_model_load(onnx.as_ptr(), onnx.len());
            let mnist = tract_model_load_with_options(
                onnx.as_ptr(),
                onnx.len(),
                0,
                1.0 / 255.0,
                0.1307,
                0.3081,
                255.0,
                shape.as_ptr(),
                4,
            );
            assert!(!plain.is_null() && !mnist.is_null());
            for idx in 0..10 {
                let rec = &samples[idx * STRIDE..(idx + 1) * STRIDE];
                let px = &rec[1..];
                let by_u8 = logits(tract_model_run_u8(
                    mnist,
                    px.as_ptr(),
                    784,
                    shape.as_ptr(),
                    4,
                ));
                assert_eq!(argmax(&by_u8), rec[0] as usize);

                // Normalizing by hand and running through the identity handle
                // must agree with the handle's own normalization.
                let manual: Vec<f32> = px
                    .iter()
                    .map(|&b| (b as f32 / 255.0 - 0.1307) / 0.3081)
                    .collect();
                let by_f32 = logits(tract_model_run_f32(
                    plain,
                    manual.as_ptr() as *const u8,
                    784 * 4,
                    shape.as_ptr(),
                    4,
                ));
                for (a, b) in by_u8.iter().zip(&by_f32) {
                    assert!((a - b).abs() < 1e-2 * (1.0 + a.abs()), "{a} vs {b}");
                }
            }
            tract_model_free(plain);
            tract_model_free(mnist);
        }
    }

    #[test]
    fn invert_flag_flips_pixels() {
        let onnx = include_bytes!("../../test/fixtures/identity.onnx");
        let shape: [usize; 1] = [3];
        let px = [0u8, 100, 255];
        unsafe {
            let m = tract_model_load_with_options(
                onnx.as_ptr(),
                onnx.len(),
                TRACT_LOAD_INVERT,
                0.5,
                0.0,
                1.0,
                255.0,
                std::ptr::null(),
                0,
            );
            assert!(!m.is_null());
            let out = logits(tract_model_run_u8(m, px.as_ptr(), 3, shape.as_ptr(), 1));
            assert_eq!(out, vec![127.5, 77.5, 0.0]);
            tract_model_free(m);
        }
    }

    #[test]
    fn model_handle_rejects_bad_arguments() {
        let onnx = include_bytes!("../../test/fixtures/identity.onnx");
        let shape: [usize; 1] = [3];
        unsafe {
            assert!(tract_model_load(std::ptr::null(), 0).is_null());
            assert!(tract_model_load(b"nope".as_ptr(), 4).is_null());
            assert!(tract_last_error_len() > 0);
            let zero_std = tract_model_load_with_options(
                onnx.as_ptr(),
                onnx.len(),
                0,
                1.0,
                0.0,
                0.0,
                255.0,
                std::ptr::null(),
                0,
            );
            assert!(zero_std.is_null());

            let m = tract_model_load(onnx.as_ptr(), onnx.len());
            let px = [1u8, 2];
            let buf = tract_model_run_u8(m, px.as_ptr(), 2, shape.as_ptr(), 1);
            assert_eq!(tract_buffer_status(buf), TRACT_STATUS_INVALID_INPUT);
            tract_buffer_free(buf);
            let buf = tract_model_run_f32(std::ptr::null(), px.as_ptr(), 2, shape.as_ptr(), 1);
            assert_eq!(tract_buffer_status(buf), TRACT_STATUS_INVALID_INPUT);
            tract_buffer_free(buf);
            tract_model_free(m);
        }
    }

    #[test]
    fn infer_bytes_cache_is_exact_for_same_length_models() {
        // ModelProto.doc_string (field 6) appended as a long string: still a
        // valid model, but big enough that a sampling key would skip bytes.
        let base = include_bytes!("../../test/fixtures/identity.onnx");
        let make = |flip_at: Option<usize>| {
            let mut doc = vec![b'a'; 100_000];
            if let Some(i) = flip_at {
                doc[i] = b'b';
            }
            let mut m = base.to_vec();
            m.extend_from_slice(&[0x32, 0xa0, 0x8d, 0x06]);
            m.extend_from_slice(&doc);
            m
        };
        let a = make(None);
        // Every position a 16k-sample, 4k head/tail fingerprint would miss.
        let b = make(Some(50_001));
        assert_eq!(a.len(), b.len());
        let input = [1.0f32, 2.0, 3.0];
        let bytes = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, 12) };
        let cached =
            |m: &[u8]| CACHED_MODEL.with(|c| c.borrow().as_ref().map(|(k, _)| k.model == m));
        unsafe {
            assert_eq!(
                logits(tract_infer_bytes(a.as_ptr(), a.len(), bytes.as_ptr(), 12)),
                input
            );
            assert_eq!(cached(&a), Some(true));
            assert_eq!(
                logits(tract_infer_bytes(b.as_ptr(), b.len(), bytes.as_ptr(), 12)),
                input
            );
            assert_eq!(
                cached(&b),
                Some(true),
                "model b must replace a, not hit a's entry"
            );
            assert_eq!(cached(&a), Some(false));
        }
    }
}
