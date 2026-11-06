"""Controlled synthetic vector-index benchmark; output goes to a caller-specified file."""
import argparse, json, platform, sys, time
from pathlib import Path
import faiss
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True); args=ap.parse_args()
    n,d,nq,k,seed,repeats=20000,128,200,10,20261107,3
    faiss.omp_set_num_threads(1)
    rng=np.random.default_rng(seed)
    base=rng.standard_normal((n,d)).astype('float32')
    query=rng.standard_normal((nq,d)).astype('float32')
    faiss.normalize_L2(base); faiss.normalize_L2(query)
    idx=faiss.IndexFlatL2(d); idx.add(base); _,truth=idx.search(query,k)
    results=[]
    def run(name, index, params, train=False):
        t=time.perf_counter()
        if train: index.train(base)
        index.add(base); build=time.perf_counter()-t
        size=len(faiss.serialize_index(index))
        index.search(query[:8], k)
        elapsed=[]; predicted=None
        for _ in range(repeats):
            t=time.perf_counter(); _,predicted=index.search(query,k); elapsed.append(time.perf_counter()-t)
        hits=sum(len(set(a)&set(b)) for a,b in zip(predicted,truth))
        results.append({'index':name,'parameters':params,'train_add_s':build,'serialized_bytes':size,
                        'recall_at_10':hits/(nq*k),'batch_elapsed_s':elapsed,
                        'batch_ms_per_query_median':float(np.median(elapsed)*1000/nq)})
        print(name,params,results[-1]['recall_at_10'],results[-1]['batch_ms_per_query_median'],flush=True)
    run('Flat',faiss.IndexFlatL2(d),{})
    run('SQ8',faiss.IndexScalarQuantizer(d,faiss.ScalarQuantizer.QT_8bit),{},True)
    run('PQ',faiss.IndexPQ(d,8,8),{'m':8,'nbits':8},True)
    for probe in (4,16,32):
        ivf=faiss.IndexIVFFlat(faiss.IndexFlatL2(d),d,64,faiss.METRIC_L2);ivf.nprobe=probe
        run('IVF-Flat',ivf,{'nlist':64,'nprobe':probe},True)
    for probe in (4,16,32):
        ivf=faiss.IndexIVFPQ(faiss.IndexFlatL2(d),d,64,8,8,faiss.METRIC_L2);ivf.nprobe=probe
        run('IVF-PQ',ivf,{'nlist':64,'nprobe':probe,'m':8,'nbits':8},True)
    for ef in (16,64,128):
        h=faiss.IndexHNSWFlat(d,16,faiss.METRIC_L2);h.hnsw.efConstruction=64;h.hnsw.efSearch=ef
        run('HNSW-Flat',h,{'M':16,'efConstruction':64,'efSearch':ef})
    data={'description':'Independent standard Gaussian vectors, row-normalized; synthetic, not a semantic retrieval dataset.',
          'n':n,'d':d,'nq':nq,'k':k,'seed':seed,'repeats':repeats,'threads':1,'metric':'squared L2',
          'faiss':faiss.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),
          'processor':platform.processor(),'results':results}
    Path(args.out).write_text(json.dumps(data,indent=2),encoding='utf8')
if __name__=='__main__': main()
