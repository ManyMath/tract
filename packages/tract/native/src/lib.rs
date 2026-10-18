//! C ABI for the `tract` package's neural-network inference.
//!
//! The Dart `FfiBackend` calls these functions. A model is loaded once
//! (`tract_onnx_model_load_*` / `tract_nnef_model_load_*`) into an opaque
//! [`TractModel`], then run any number of times with a set of input tensors,
//! producing a set of output tensors. Tensors cross the boundary as opaque
//! [`TractValue`] handles carrying a datum type, a shape and a contiguous,
//! native-endian data buffer.
//!
//! Errors are reported out-of-band: fallible calls return null / a negative
//! count and stash a message retrievable with [`tract_last_error`].

// The exported functions take opaque handles produced by this library; the
// caller is responsible for their validity, which makes the raw-pointer derefs
// expected here rather than a smell.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::model::TypedRunnableModel;

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

/// Returns this library's version as a static NUL-terminated string.
///
/// The Dart `FfiBackend` also calls it to check that the library is present.
#[no_mangle]
pub extern "C" fn tract_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
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
    }
}
