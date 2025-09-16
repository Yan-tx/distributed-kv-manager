use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// KV数据结构定义
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct KVData {
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    hidden: Option<Vec<u8>>,
    input_tokens: Vec<u8>,
    roi: Vec<u8>,
    k_shape: Vec<usize>,
    v_shape: Vec<usize>,
    hidden_shape: Option<Vec<usize>>,
}

#[pymodule]
fn kv_serializer(_py: Python, m: &PyModule) -> PyResult<()> {
    /// 打包KV数据为字节流
    #[pyfunction]
    #[pyo3(signature = (_k_cache, _v_cache, _hidden=None, _input_tokens=None, _roi=None))]
    fn pack_kv_data(
        _py: Python,
        _k_cache: &PyDict,
        _v_cache: &PyDict,
        _hidden: Option<&PyDict>,
        _input_tokens: Option<&PyDict>,
        _roi: Option<&PyDict>,
    ) -> PyResult<Vec<u8>> {
        // 简化实现 - 直接从PyDict中提取数据
        // 在完整实现中，这里需要处理PyTorch张量
        
        let k_cache_data = vec![1u8; 100]; // 示例数据
        let v_cache_data = vec![2u8; 100]; // 示例数据
        let hidden_data = Some(vec![3u8; 50]); // 示例数据
        let input_tokens_data = vec![4u8; 20]; // 示例数据
        let roi_data = vec![5u8; 10]; // 示例数据
        
        // 创建KVData结构
        let kv_data = KVData {
            k_cache: k_cache_data,
            v_cache: v_cache_data,
            hidden: hidden_data,
            input_tokens: input_tokens_data,
            roi: roi_data,
            k_shape: vec![10, 10],
            v_shape: vec![10, 10],
            hidden_shape: Some(vec![5, 10]),
        };
        
        // 使用bincode序列化
        let serialized = bincode::serialize(&kv_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("序列化失败: {}", e)))?;
        
        Ok(serialized)
    }

    /// 从字节流解包KV数据
    #[pyfunction]
    fn unpack_kv_data(py: Python, data: Vec<u8>) -> PyResult<(PyObject, PyObject, PyObject)> {
        // 使用bincode反序列化
        let kv_data: KVData = bincode::deserialize(&data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("反序列化失败: {}", e)))?;
        
        // 创建PyBytes对象
        let k_cache = PyBytes::new(py, &kv_data.k_cache).into();
        let v_cache = PyBytes::new(py, &kv_data.v_cache).into();
        let hidden = match &kv_data.hidden {
            Some(h_data) => PyBytes::new(py, h_data).into(),
            None => py.None(),
        };
        
        Ok((k_cache, v_cache, hidden))
    }
    
    // 添加函数到模块
    m.add_function(wrap_pyfunction!(pack_kv_data)(m)?)?;
    m.add_function(wrap_pyfunction!(unpack_kv_data)(m)?)?;

    Ok(())
}