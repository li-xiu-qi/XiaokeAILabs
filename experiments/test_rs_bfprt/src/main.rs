use std::time::Instant;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use plotly::{Plot, Scatter};
use plotly::common::{Mode, Title};
use plotly::layout::{Axis, Layout};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    size: usize,
    bfprt_ns: u128,
    quickselect_ns: u128,
    quickselect_improved_ns: u128,
    heap_select_ns: u128,
    sort_select_ns: u128,
    std_select_ns: u128,
}

/// 插入排序，用于小数组
fn insertion_sort(arr: &mut [i32]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i as i32 - 1;
        
        while j >= 0 && arr[j as usize] > key {
            arr[(j + 1) as usize] = arr[j as usize];
            j -= 1;
        }
        arr[(j + 1) as usize] = key;
    }
}

/// 找到中位数的中位数
fn find_median_of_medians(arr: &[i32]) -> i32 {
    if arr.len() <= 5 {
        let mut small_arr = arr.to_vec();
        small_arr.sort_unstable();
        return small_arr[small_arr.len() / 2];
    }
    
    let mut medians = Vec::new();
    for chunk in arr.chunks(5) {
        let mut group = chunk.to_vec();
        group.sort_unstable();
        medians.push(group[group.len() / 2]);
    }
    
    find_median_of_medians(&medians)
}

/// 三路快速分区
fn three_way_partition(arr: &[i32], pivot: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut less = Vec::new();
    let mut equal = Vec::new();
    let mut greater = Vec::new();
    
    for &item in arr {
        match item.cmp(&pivot) {
            std::cmp::Ordering::Less => less.push(item),
            std::cmp::Ordering::Equal => equal.push(item),
            std::cmp::Ordering::Greater => greater.push(item),
        }
    }
    
    (less, equal, greater)
}

/// BFPRT算法 - 保证O(n)最坏情况
fn bfprt_select(arr: &[i32], k: usize) -> i32 {
    if arr.len() == 1 {
        return arr[0];
    }
    
    let pivot = find_median_of_medians(arr);
    let (less, equal, greater) = three_way_partition(arr, pivot);
    
    if k <= less.len() {
        bfprt_select(&less, k)
    } else if k <= less.len() + equal.len() {
        pivot
    } else {
        bfprt_select(&greater, k - less.len() - equal.len())
    }
}

/// 快速选择算法 - 平均O(n)，最坏O(n²)
fn quickselect(arr: &[i32], k: usize) -> i32 {
    if arr.len() == 1 {
        return arr[0];
    }
    
    let mut rng = thread_rng();
    let pivot = arr[rng.gen_range(0..arr.len())];
    
    let (less, equal, greater) = three_way_partition(arr, pivot);
    
    if k <= less.len() {
        quickselect(&less, k)
    } else if k <= less.len() + equal.len() {
        pivot
    } else {
        quickselect(&greater, k - less.len() - equal.len())
    }
}

/// 改进的快速选择 - 使用三数取中
fn quickselect_improved(arr: &[i32], k: usize) -> i32 {
    if arr.len() == 1 {
        return arr[0];
    }
    
    // 三数取中选择pivot
    let pivot = if arr.len() >= 3 {
        let mut rng = thread_rng();
        let a = rng.gen_range(0..arr.len());
        let b = rng.gen_range(0..arr.len());
        let c = rng.gen_range(0..arr.len());
        
        let va = arr[a];
        let vb = arr[b];
        let vc = arr[c];
        
        if (va <= vb && vb <= vc) || (vc <= vb && vb <= va) {
            vb
        } else if (vb <= va && va <= vc) || (vc <= va && va <= vb) {
            va
        } else {
            vc
        }
    } else {
        arr[0]
    };
    
    let (less, equal, greater) = three_way_partition(arr, pivot);
    
    if k <= less.len() {
        quickselect_improved(&less, k)
    } else if k <= less.len() + equal.len() {
        pivot
    } else {
        quickselect_improved(&greater, k - less.len() - equal.len())
    }
}

/// 堆选择算法 - O(n + k log n)
fn heap_select(arr: &[i32], k: usize) -> i32 {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    
    if k <= arr.len() / 2 {
        // 使用最小堆找第k小
        let mut heap = BinaryHeap::new();
        for &item in arr.iter().take(k) {
            heap.push(item);
        }
        
        for &item in arr.iter().skip(k) {
            if item < *heap.peek().unwrap() {
                heap.pop();
                heap.push(item);
            }
        }
        
        *heap.peek().unwrap()
    } else {
        // 使用最大堆找第(n-k+1)大
        let target = arr.len() - k + 1;
        let mut heap = BinaryHeap::new();
        for &item in arr.iter().take(target) {
            heap.push(Reverse(item));
        }
        
        for &item in arr.iter().skip(target) {
            if item > heap.peek().unwrap().0 {
                heap.pop();
                heap.push(Reverse(item));
            }
        }
        
        heap.peek().unwrap().0
    }
}

/// 使用标准库排序
fn sort_select(arr: &[i32], k: usize) -> i32 {
    let mut sorted_arr = arr.to_vec();
    sorted_arr.sort_unstable();
    sorted_arr[k - 1]
}

/// 使用标准库的select_nth_unstable
fn std_select(arr: &[i32], k: usize) -> i32 {
    let mut arr_copy = arr.to_vec();
    arr_copy.select_nth_unstable(k - 1);
    arr_copy[k - 1]
}

fn main() {
    println!("Rust选择算法性能对比测试");
    println!("============================");
    
    let test_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000];
    let num_trials = 10;
    let k_ratio = 0.3;
    
    println!("测试配置:");
    println!("  数组大小: {:?}", test_sizes);
    println!("  每个大小测试次数: {}", num_trials);
    println!("  k值比例: {}", k_ratio);
    println!();
    
    let mut results = Vec::new();
    
    for &size in &test_sizes {
        println!("测试数组大小: {}", size);
        let k = (size as f64 * k_ratio) as usize;
        
        let mut bfprt_total = 0u128;
        let mut quickselect_total = 0u128;
        let mut quickselect_improved_total = 0u128;
        let mut heap_total = 0u128;
        let mut sort_total = 0u128;
        let mut std_total = 0u128;
        
        for _ in 0..num_trials {
            // 生成随机数组
            let mut rng = thread_rng();
            let arr: Vec<i32> = (0..size).map(|_| rng.gen_range(1..size as i32 * 10)).collect();
            
            // 测试BFPRT
            let start = Instant::now();
            let _ = bfprt_select(&arr, k);
            bfprt_total += start.elapsed().as_nanos();
            
            // 测试快速选择
            let start = Instant::now();
            let _ = quickselect(&arr, k);
            quickselect_total += start.elapsed().as_nanos();
            
            // 测试改进快速选择
            let start = Instant::now();
            let _ = quickselect_improved(&arr, k);
            quickselect_improved_total += start.elapsed().as_nanos();
            
            // 测试堆选择
            let start = Instant::now();
            let _ = heap_select(&arr, k);
            heap_total += start.elapsed().as_nanos();
            
            // 测试排序
            let start = Instant::now();
            let _ = sort_select(&arr, k);
            sort_total += start.elapsed().as_nanos();
            
            // 测试标准库
            let start = Instant::now();
            let _ = std_select(&arr, k);
            std_total += start.elapsed().as_nanos();
        }
        
        let avg_bfprt = bfprt_total / num_trials as u128;
        let avg_quickselect = quickselect_total / num_trials as u128;
        let avg_quickselect_improved = quickselect_improved_total / num_trials as u128;
        let avg_heap = heap_total / num_trials as u128;
        let avg_sort = sort_total / num_trials as u128;
        let avg_std = std_total / num_trials as u128;
        
        // 保存结果用于可视化
        results.push(BenchmarkResult {
            size,
            bfprt_ns: avg_bfprt,
            quickselect_ns: avg_quickselect,
            quickselect_improved_ns: avg_quickselect_improved,
            heap_select_ns: avg_heap,
            sort_select_ns: avg_sort,
            std_select_ns: avg_std,
        });
        
        println!("  BFPRT算法        : {:>8} ns", avg_bfprt);
        println!("  快速选择算法      : {:>8} ns", avg_quickselect);
        println!("  改进快速选择      : {:>8} ns", avg_quickselect_improved);
        println!("  堆选择算法        : {:>8} ns", avg_heap);
        println!("  排序算法         : {:>8} ns", avg_sort);
        println!("  标准库算法        : {:>8} ns", avg_std);
        
        // 计算相对性能
        let times = [avg_bfprt, avg_quickselect, avg_quickselect_improved, avg_heap, avg_sort, avg_std];
        let fastest = *times.iter().min().unwrap();
        println!("  相对性能比较 (以最快为1.0x):");
        println!("    BFPRT        : {:.2}x", avg_bfprt as f64 / fastest as f64);
        println!("    快速选择      : {:.2}x", avg_quickselect as f64 / fastest as f64);
        println!("    改进快速选择   : {:.2}x", avg_quickselect_improved as f64 / fastest as f64);
        println!("    堆选择       : {:.2}x", avg_heap as f64 / fastest as f64);
        println!("    排序算法      : {:.2}x", avg_sort as f64 / fastest as f64);
        println!("    标准库算法     : {:.2}x", avg_std as f64 / fastest as f64);
        println!();
    }
    
    // 生成可视化图表
    println!("正在生成性能可视化图表...");
    if let Err(e) = create_performance_charts(&results) {
        eprintln!("生成图表时出错: {}", e);
    } else {
        println!("性能图表已保存到 performance_chart.html 和 relative_performance.html");
    }

    // 保存结果到JSON文件
    if let Err(e) = save_results_to_json(&results) {
        eprintln!("保存JSON数据时出错: {}", e);
    } else {
        println!("性能数据已保存到 benchmark_results.json");
    }
    
    // 正确性测试
    println!("正确性验证:");
    let test_arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let k = 5;
    
    let result_bfprt = bfprt_select(&test_arr, k);
    let result_quickselect = quickselect(&test_arr, k);
    let result_quickselect_improved = quickselect_improved(&test_arr, k);
    let result_heap = heap_select(&test_arr, k);
    let result_sort = sort_select(&test_arr, k);
    let result_std = std_select(&test_arr, k);
    
    println!("  测试数组: {:?}", test_arr);
    println!("  查找第{}小的元素", k);
    println!("  BFPRT结果       : {}", result_bfprt);
    println!("  快速选择结果     : {}", result_quickselect);
    println!("  改进快速选择结果  : {}", result_quickselect_improved);
    println!("  堆选择结果       : {}", result_heap);
    println!("  排序结果        : {}", result_sort);
    println!("  标准库结果       : {}", result_std);
    
    let mut sorted_test = test_arr.clone();
    sorted_test.sort();
    let expected = sorted_test[k - 1];
    println!("  预期结果        : {}", expected);
    
    let all_correct = [result_bfprt, result_quickselect, result_quickselect_improved, 
                      result_heap, result_sort, result_std]
                      .iter().all(|&r| r == expected);
    println!("  所有算法结果正确: {}", all_correct);
}

/// 使用plotly.rs创建性能对比图表
fn create_performance_charts(results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    let sizes: Vec<usize> = results.iter().map(|r| r.size).collect();
    
    // 创建绝对性能图表
    let mut plot = Plot::new();
    
    // BFPRT算法
    let bfprt_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.bfprt_ns as f64).collect::<Vec<f64>>()
    )
    .name("BFPRT算法")
    .mode(Mode::LinesMarkers);
    plot.add_trace(bfprt_trace);
    
    // 快速选择
    let quickselect_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.quickselect_ns as f64).collect::<Vec<f64>>()
    )
    .name("快速选择")
    .mode(Mode::LinesMarkers);
    plot.add_trace(quickselect_trace);
    
    // 改进快速选择
    let improved_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.quickselect_improved_ns as f64).collect::<Vec<f64>>()
    )
    .name("改进快速选择")
    .mode(Mode::LinesMarkers);
    plot.add_trace(improved_trace);
    
    // 堆选择
    let heap_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.heap_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("堆选择")
    .mode(Mode::LinesMarkers);
    plot.add_trace(heap_trace);
    
    // 排序算法
    let sort_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.sort_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("排序算法")
    .mode(Mode::LinesMarkers);
    plot.add_trace(sort_trace);
    
    // 标准库
    let std_trace = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("标准库算法")
    .mode(Mode::LinesMarkers);
    plot.add_trace(std_trace);
    
    let layout = Layout::new()
        .title(Title::new("Rust选择算法性能对比"))
        .x_axis(Axis::new().title(Title::new("数组大小")))
        .y_axis(Axis::new().title(Title::new("执行时间 (纳秒)")));
    plot.set_layout(layout);
    
    plot.write_html("performance_chart.html");
    
    // 创建相对性能图表
    let mut relative_plot = Plot::new();
    
    // 相对于标准库的性能比较
    let bfprt_relative = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.bfprt_ns as f64 / r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("BFPRT vs 标准库")
    .mode(Mode::LinesMarkers);
    relative_plot.add_trace(bfprt_relative);
    
    let quick_relative = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.quickselect_ns as f64 / r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("快速选择 vs 标准库")
    .mode(Mode::LinesMarkers);
    relative_plot.add_trace(quick_relative);
    
    let improved_relative = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.quickselect_improved_ns as f64 / r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("改进快速选择 vs 标准库")
    .mode(Mode::LinesMarkers);
    relative_plot.add_trace(improved_relative);
    
    let heap_relative = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.heap_select_ns as f64 / r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("堆选择 vs 标准库")
    .mode(Mode::LinesMarkers);
    relative_plot.add_trace(heap_relative);
    
    let sort_relative = Scatter::new(
        sizes.clone(),
        results.iter().map(|r| r.sort_select_ns as f64 / r.std_select_ns as f64).collect::<Vec<f64>>()
    )
    .name("排序算法 vs 标准库")
    .mode(Mode::LinesMarkers);
    relative_plot.add_trace(sort_relative);
    
    // 基准线
    let baseline = Scatter::new(
        sizes.clone(),
        vec![1.0; sizes.len()]
    )
    .name("标准库基准线")
    .mode(Mode::Lines);
    relative_plot.add_trace(baseline);
    
    let relative_layout = Layout::new()
        .title(Title::new("相对性能对比 (以标准库为基准)"))
        .x_axis(Axis::new().title(Title::new("数组大小")))
        .y_axis(Axis::new().title(Title::new("相对性能倍数")));
    relative_plot.set_layout(relative_layout);
    
    relative_plot.write_html("relative_performance.html");
    
    Ok(())
}

/// 保存结果到JSON文件
fn save_results_to_json(results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write("benchmark_results.json", json)?;
    Ok(())
}
