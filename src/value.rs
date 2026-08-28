//! Recursive Python -> `valuable::Valuable` value tree.
//!
//! Dispatch order mirrors `orjson`: bool before int (bool is a subclass of
//! int), then int / float / str / list / tuple / dict / array / dataclass,
//! with a `str(obj)` fallback so the call never raises on exotic inputs.

use pyo3::prelude::*;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyString, PyTuple,
};
use valuable::{Mappable, Valuable, Value, Visit};

pub enum PyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    Map(OrderedMap),
}

/// Ordered string-keyed map. Preserves insertion order from `dict.iter()` so
/// downstream consumers (formatters, JSON exporters) see kwargs in the order
/// the user wrote them.
pub struct OrderedMap(pub Vec<(String, PyValue)>);

impl Valuable for PyValue {
    fn as_value(&self) -> Value<'_> {
        match self {
            PyValue::Null => Value::Unit,
            PyValue::Bool(b) => Value::Bool(*b),
            PyValue::Int(i) => Value::I64(*i),
            PyValue::Float(f) => Value::F64(*f),
            PyValue::Str(s) => Value::String(s),
            PyValue::List(l) => Value::Listable(l),
            PyValue::Map(m) => Value::Mappable(m),
        }
    }

    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            PyValue::List(l) => visit.visit_value(Value::Listable(l)),
            PyValue::Map(m) => visit.visit_value(Value::Mappable(m)),
            other => visit.visit_value(other.as_value()),
        }
    }
}

impl Valuable for OrderedMap {
    fn as_value(&self) -> Value<'_> {
        Value::Mappable(self)
    }

    fn visit(&self, visit: &mut dyn Visit) {
        for (k, v) in &self.0 {
            visit.visit_entry(Value::String(k), v.as_value());
        }
    }
}

impl Mappable for OrderedMap {
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<PyValue> {
    if obj.is_none() {
        return Ok(PyValue::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(PyValue::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(PyValue::Int(i));
        }
        if let Ok(f) = obj.extract::<f64>() {
            return Ok(PyValue::Float(f));
        }
        return Ok(PyValue::Str(obj.str()?.to_string()));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(PyValue::Float(obj.extract::<f64>()?));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(PyValue::Str(s.to_string()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(PyValue::List(out));
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(PyValue::List(out));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut out = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key: String = k.extract().or_else(|_| k.str()?.extract())?;
            out.push((key, py_to_value(&v)?));
        }
        return Ok(PyValue::Map(OrderedMap(out)));
    }
    // numpy/torch/jax-style arrays and scalars — detected via the `__array__`
    // protocol so we don't need numpy as a build dep. `.tolist()` returns
    // nested Python lists for ndim>=1 and a native Python scalar for 0-d
    // arrays / numpy scalars; recursing handles both shapes.
    if obj.hasattr("__array__")? {
        let as_list = obj.call_method0("tolist")?;
        return py_to_value(&as_list);
    }
    // Dataclass instance: walk dataclasses.fields(obj) and pull each attr.
    if obj.hasattr("__dataclass_fields__")? {
        let dataclasses = obj.py().import("dataclasses")?;
        let fields = dataclasses.call_method1("fields", (obj,))?;
        let mut out = Vec::new();
        for field in fields.try_iter()? {
            let field = field?;
            let name: String = field.getattr("name")?.extract()?;
            let val = obj.getattr(name.as_str())?;
            out.push((name, py_to_value(&val)?));
        }
        return Ok(PyValue::Map(OrderedMap(out)));
    }
    Ok(PyValue::Str(obj.str()?.to_string()))
}
