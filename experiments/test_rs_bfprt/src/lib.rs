use rand::prelude::*;

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
pub fn bfprt_select(arr: &[i32], k: usize) -> i32 {
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
pub fn quickselect(arr: &[i32], k: usize) -> i32 {
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
pub fn quickselect_improved(arr: &[i32], k: usize) -> i32 {
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
pub fn heap_select(arr: &[i32], k: usize) -> i32 {
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
pub fn sort_select(arr: &[i32], k: usize) -> i32 {
    let mut sorted_arr = arr.to_vec();
    sorted_arr.sort_unstable();
    sorted_arr[k - 1]
}

/// 使用标准库的select_nth_unstable
pub fn std_select(arr: &[i32], k: usize) -> i32 {
    let mut arr_copy = arr.to_vec();
    arr_copy.select_nth_unstable(k - 1);
    arr_copy[k - 1]
}
