# Visual document retrieval with ColSmol

Finding the page that answers a question in reports and tables often requires OCR and a text index. Visual document retrieval offers a different route: encode the page as an image and match the question against its visual representation. We tested the roughly 256M-parameter ColSmol model on 100 held-out DocVQA page candidates.

![From questions and document pages to late-interaction ranking](retrieval-flow.svg)

On DGX Spark, Recall@1 was **0.55**, Recall@5 **0.78**, and nDCG@5 **0.6664**. These are results for this small fixed candidate set, not an official full ViDoRe score.

## How it works

ColSmol produces multiple vectors for each question and page. ColBERT-style late interaction finds the most similar page vector for each query vector and sums those similarities:

$$s(q,d)=\sum_i\max_j \widehat{q_i}^{\mathsf T}\widehat{d_j}.$$

This allows different parts of a question to match different page features, at the cost of storing and comparing multiple vectors. The experiment does not measure full-corpus indexing or online serving latency. See the [ColPali paper](https://arxiv.org/abs/2407.01449) and [model card](https://huggingface.co/vidore/colSmol-256M).

## Data and results

The [ViDoRe DocVQA test subset](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) pairs document-page images with questions. In dataset order, we selected the first 100 distinct `image_filename` values and the first associated question for each page. Each query is ranked against all 100 pages; its paired page is treated as the only known positive.

| Metric | Result | Interpretation |
|---|---:|---|
| Recall@1 | 0.55 | Known paired page ranked first |
| Recall@5 | 0.78 | Known paired page ranked in top five |
| nDCG@5 | 0.6664 | Gives more credit to earlier ranks |

Other pages may also be relevant but unlabeled. The 100-page candidate pool cannot establish full-corpus retrieval quality. The pretrained model was evaluated without further training. Its model card reports that the original training set did not share multipage PDFs with ViDoRe; this does not rule out overlap in backbone pretraining.

[baseline100.json](baseline100.json) contains anonymous per-query positive ranks, with no images, question text, or original dataset IDs. `python recompute.py` checks the aggregate metrics from those ranks; it cannot independently establish the underlying model scores or relevance labels.

## Run the baseline

The run used Linux on NVIDIA GB10, BF16, PyTorch 2.14.0+cu130, Transformers 4.56.2, ColPali Engine 0.3.13, PEFT 0.17.1 and Datasets 4.1.1. Fetch the data and model weights from their original sources:

```bash
python -m pip install 'transformers==4.56.2' 'colpali-engine==0.3.13' 'peft==0.17.1' 'datasets==4.1.1' huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('vidore/colSmol-256M', revision='c79b633e17e060cc109b11aa2bf92a1517d3bd6f', local_dir='model')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('vidore/ColSmolVLM-Instruct-256M-base', revision='99ca96f1f6b95b3a69e6abef74a2416cb738fed0', local_dir='base')"
python prepare_data.py --output data
python baseline.py --data data --base base --model model --out /tmp/visual-smoke.json --count 10
python baseline.py --data data --base base --model model --out /tmp/visual-baseline100.json --count 100
```

The published checkpoint is an adapter: load the specified base first, then the adapter. The base alone is not a trained retriever. The run took 78.12 seconds for encoding and pairwise scoring, excluding model and data loading; peak PyTorch allocation was approximately 2.81 GiB, not total system memory. The script retains a 50 GiB available-memory guard used on this machine; it is not a general model requirement.

The model card notes that current ColPali Engine prompting differs from the original training format. This run used version 0.3.13, so its result should not be compared directly with the model-card score. Software, hardware and floating-point changes can affect ranks. The 10-page run only checks the environment; the result above comes from 100 pages.

## Licensing and future work

The model and base repository metadata list MIT; the base card describes the backbone as Apache-2.0 and adapters as MIT. The model card also contains a ColQwen2 licensing paragraph, so verify the applicable files and upstream terms before redistributing weights. The ViDoRe card lists MIT; observe original DocVQA and underlying document terms as applicable. No dataset pages, question text or model weights are redistributed here.

Fine-tuning and distillation have not yet been run. A useful follow-up would fix a training set separated from test documents by source, then compare the pretrained model, equal-budget supervised training and teacher-score distillation on held-out documents.
