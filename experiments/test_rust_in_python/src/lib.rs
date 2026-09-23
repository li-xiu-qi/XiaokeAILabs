use pyo3::prelude::{PyResult, Python, PyModule, pyfunction, pymodule, wrap_pyfunction};

#[pyfunction]
fn add_numbers(a: i64, b: i64) -> PyResult<i64> {
    Ok(a + b)
}

// 演示如何解除GIL的函数
#[pyfunction]
fn compute_sum_without_gil(py: Python, numbers: Vec<i64>) -> PyResult<i64> {
    // 使用allow_threads方法暂时释放GIL
    // 这允许其他Python线程在此Rust代码执行期间运行
    py.allow_threads(|| {
        // 在此闭包中，不能访问任何Python对象
        // 但可以执行计算密集型操作
        
        // 模拟一个耗时的计算
        let mut sum = 0;
        for num in numbers {
            sum += num;
            // 模拟计算耗时
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        
        // 返回计算结果
        Ok(sum)
    })
}

#[pymodule]
fn rust_adder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_without_gil, m)?)?;
    Ok(())
}
