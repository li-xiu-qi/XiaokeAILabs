use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::prelude::*;
use test_rs_bfprt::*;

fn benchmark_selection_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection_algorithms");
    
    // 测试不同的数组大小
    let sizes = [100, 500, 1000, 5000, 10000];
    
    for size in sizes.iter() {
        let mut rng = thread_rng();
        let arr: Vec<i32> = (0..*size).map(|_| rng.gen_range(1..*size * 10)).collect();
        let k = (*size as f64 * 0.3) as usize; // 第30%位置的元素
        
        group.bench_with_input(BenchmarkId::new("bfprt", size), size, |b, _| {
            b.iter(|| bfprt_select(black_box(&arr), black_box(k)))
        });
        
        group.bench_with_input(BenchmarkId::new("quickselect", size), size, |b, _| {
            b.iter(|| quickselect(black_box(&arr), black_box(k)))
        });
        
        group.bench_with_input(BenchmarkId::new("quickselect_improved", size), size, |b, _| {
            b.iter(|| quickselect_improved(black_box(&arr), black_box(k)))
        });
        
        group.bench_with_input(BenchmarkId::new("heap_select", size), size, |b, _| {
            b.iter(|| heap_select(black_box(&arr), black_box(k)))
        });
        
        group.bench_with_input(BenchmarkId::new("sort_select", size), size, |b, _| {
            b.iter(|| sort_select(black_box(&arr), black_box(k)))
        });
        
        group.bench_with_input(BenchmarkId::new("std_select", size), size, |b, _| {
            b.iter(|| std_select(black_box(&arr), black_box(k)))
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_selection_algorithms);
criterion_main!(benches);
