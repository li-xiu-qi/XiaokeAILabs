---
library_name: transformers
license: cc-by-nc-4.0
tags:
- xlm-roberta
- eva02
- clip
- feature-extraction
- sentence-similarity
- retrieval
- multimodal
- multi-modal
- crossmodal
- cross-modal
- mteb
- clip-benchmark
- vidore
- transformers
- sentence-transformers
- onnx
- safetensors
- transformers.js
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- id
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- 'no'
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
inference: false
base_model:
- jinaai/xlm-roberta-flash-implementation
---

<br><br>

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>


<p align="center">
<b>The embedding set trained by <a href="https://jina.ai/"><b>Jina AI</b></a>.</b>
</p>

<p align="center">
<b>Jina CLIP v2: Multilingual Multimodal Embeddings for Texts and Images</b>
</p>


## Quick Start

[Blog](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images) | [Technical Report](https://arxiv.org/abs/2412.08802) | [Azure](https://azuremarketplace.microsoft.com/en-gb/marketplace/apps/jinaai.jina-clip-v2-vm?tab=Overview) | [AWS SageMaker](https://aws.amazon.com/marketplace/pp/prodview-bfbctuqmky676) | [Google Cloud Platform](https://console.cloud.google.com/marketplace/browse?hl=en&inv=1&invt=AbiD-g&q=jina) | [API](https://jina.ai/embeddings)


## Intended Usage & Model Info

`jina-clip-v2` is a **general-purpose multilingual multimodal embedding model for text & images**.

Multimodal embeddings enable searching and understanding data across different modalities through a coherent representation. They serve as the backbone of neural information retrieval and multimodal GenAI applications.

Built upon [`jina-clip-v1`](https://huggingface.co/jinaai/jina-clip-v1) and our recently released [`jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3), `jina-clip-v2` features several significant improvements:

* **Improved Performance**: v2 shows a 3% performance improvement over v1 in both text-image and text-text retrieval tasks. Similar to v1, v2's text encoder can serve as an effective multilingual long-context dense retriever. It performs on par with our frontier model `jina-embeddings-v3` (currently the best multilingual embeddings under 1B parameters on MTEB).
* **Multilingual Support**: Using the same backbone as `jina-embeddings-v3` for the text tower, `jina-clip-v2` supports 89 languages for multilingual-image retrieval, showing up to 4% improvement compared to `nllb-clip-large-siglip` on multilingual image retrieval tasks.
* **Higher Image Resolution**: v2 now supports 512x512 input image resolution, a significant increase from v1's 224x224. This higher resolution enables better processing of detailed images, improved feature extraction, and more accurate recognition of fine-grained visual elements.
* **Matryoshka Representations**: v2 allows users to truncate the output dimensions of both text and image embeddings from 1024 down to 64, reducing storage and processing overhead while maintaining strong performance.

Measuring 0.9B parameters, `jina-clip-v2` combines two powerful encoders:
* the text encoder `Jina-XLM-RoBERTa` (the backbone of `jina-embeddings-v3`) and 
* the vision encoder `EVA02-L14` (an efficient vision Transformer developed by BAAI).

| FEATURE               | TEXT ENCODER            | IMAGE ENCODER    |
|-----------------------|-------------------------|------------------|
| Base Model	           | Jina-XLM-RoBERTa	       | EVA02-L          |
| Parameters	           | 561M                    | 304M             |
| Input Specification	  | 8,192 tokens (max)	     | 512×512 pixels   |
| Min Output Dimensions | 64                      | 64               |
| Max Output Dimensions | 1,024                   | 1,024            |
| Layers	               | 24                      | 24               |
| Attention Mechanism	  | FlashAttention2	        | xFormers         |
| Pooling Strategy	     | Mean pooling	           | CLS pooling      |
| Additional Features	  | 89 languages supported	 | Patch size 14x14 |


These encoders are jointly trained to create aligned representations of images and text.

CLIP-like models have established themselves as the backbone for general-purpose multimodal applications. With `jina-clip-v2`, we're taking these capabilities to the next level, breaking down language barriers to deliver more accurate cross-modal understanding and retrieval. We're confident this release delivers a promise in making multimodal search and retrieval both more powerful and more accessible to developers worldwide.



## Training, Data, Parameters

Please refer to our [technical report of jina-clip-v2](https://arxiv.org/abs/2412.08802) for the model and training details.

[technical report of jina-clip-v1](https://arxiv.org/abs/2405.20204)

## Faster Inference: FA2, XFormers and bf16

On a CUDA enabled torch environment, the model comes in `torch.bfloat16` 
precision by default. It is highly recommended to install 
[FlashAttention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) 
and [xFormers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) 
to make use of their efficient attention mechanism implementations.


## Usage

<details>
  <summary>via Jina AI <a href="https://jina.ai/embeddings/">Embedding API</a></summary>

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer [JINA_AI_API_TOKEN]" \
  -d @- <<EOFEOF
  {
    "model": "jina-clip-v2",
    "dimensions": 1024,
    "task": "retrieval.query",
    "normalized": true,
    "embedding_type": "float",
    "input": [
        {
            "text": "غروب جميل على الشاطئ"
        },
        {
            "text": "海滩上美丽的日落"
        },
        {
            "text": "A beautiful sunset over the beach"
        },
        {
            "text": "Un beau coucher de soleil sur la plage"
        },
        {
            "text": "Ein wunderschöner Sonnenuntergang am Strand"
        },
        {
            "text": "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία"
        },
        {
            "text": "समुद्र तट पर एक खूबसूरत सूर्यास्त"
        },
        {
            "text": "Un bellissimo tramonto sulla spiaggia"
        },
        {
            "text": "浜辺に沈む美しい夕日"
        },
        {
            "text": "해변 위로 아름다운 일몰"
        },
        {
            "image": "https://i.ibb.co/nQNGqL0/beach1.jpg"
        },
        {
            "image": "https://i.ibb.co/r5w8hG8/beach2.jpg"
        }
    ]
  }
EOFEOF
```

</details>

<details>
  <summary>via <a href="https://huggingface.co/docs/transformers/en/index">transformers</a></summary>

```python
# !pip install transformers einops timm pillow
from transformers import AutoModel

# Initialize the model
model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Choose a matryoshka dimension, set to None to get the full 1024-dim vectors
truncate_dim = 512

# Encode text and images
text_embeddings = model.encode_text(sentences, truncate_dim=truncate_dim)
image_embeddings = model.encode_image(
    image_urls, truncate_dim=truncate_dim
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode_text(
    query, task='retrieval.query', truncate_dim=truncate_dim
)

# Text to Image
print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
# Image to Image
print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
# Text to Text
print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))
```
</details>

<details>
  <summary>via <a href="https://sbert.net/">sentence-transformers</a></summary>
  
```python
# !pip install sentence-transformers einops timm pillow
from sentence_transformers import SentenceTransformer

# Choose a matryoshka dimension
truncate_dim = 512

# Initialize the model
model = SentenceTransformer(
    'jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=truncate_dim
)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Encode text and images
text_embeddings = model.encode(sentences, normalize_embeddings=True)
image_embeddings = model.encode(
    image_urls, normalize_embeddings=True
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode(
    query, prompt_name='retrieval.query', normalize_embeddings=True
)  
```
</details>

<details>
  <summary>via <a href="https://huggingface.co/docs/transformers.js/en/index">transformers.js</a></summary>

> [!NOTE]
> JinaCLIP was added in Transformers.js v3.1.0, so make sure you're using a compatible version!
> See the [release notes](https://github.com/huggingface/transformers.js/releases/tag/3.1.0) for more information.

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Compute text and/or image embeddings with `jinaai/jina-clip-v2`:
```js
import { AutoModel, AutoProcessor, RawImage, matmul } from "@huggingface/transformers";

// Load processor and model
const model_id = "jinaai/jina-clip-v2";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id, { dtype: "q4" /* e.g., "fp16", "q8", or "q4" */ });

// Prepare inputs
const urls = ["https://i.ibb.co/nQNGqL0/beach1.jpg", "https://i.ibb.co/r5w8hG8/beach2.jpg"];
const images = await Promise.all(urls.map(url => RawImage.read(url)));
const sentences = [
    "غروب جميل على الشاطئ", // Arabic
    "海滩上美丽的日落", // Chinese
    "Un beau coucher de soleil sur la plage", // French
    "Ein wunderschöner Sonnenuntergang am Strand", // German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία", // Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त", // Hindi
    "Un bellissimo tramonto sulla spiaggia", // Italian
    "浜辺に沈む美しい夕日", // Japanese
    "해변 위로 아름다운 일몰", // Korean
];

// Encode text and images
const inputs = await processor(sentences, images, { padding: true, truncation: true });
const { l2norm_text_embeddings, l2norm_image_embeddings } = await model(inputs);

// Encode query (text-only)
const query_prefix = "Represent the query for retrieving evidence documents: ";
const query_inputs = await processor(query_prefix + "beautiful sunset over the beach");
const { l2norm_text_embeddings: query_embeddings } = await model(query_inputs);

// Compute text-image similarity scores
const text_to_image_scores = await matmul(query_embeddings, l2norm_image_embeddings.transpose(1, 0));
console.log("text-image similarity scores", text_to_image_scores.tolist()[0]); // [0.29530206322669983, 0.3183615803718567]

// Compute image-image similarity scores
const image_to_image_score = await matmul(l2norm_image_embeddings[0], l2norm_image_embeddings[1]);
console.log("image-image similarity score", image_to_image_score.item()); // 0.9344457387924194

// Compute text-text similarity scores
const text_to_text_scores = await matmul(query_embeddings, l2norm_text_embeddings.transpose(1, 0));
console.log("text-text similarity scores", text_to_text_scores.tolist()[0]); // [0.5566609501838684, 0.7028406858444214, 0.582255482673645, 0.6648036241531372, 0.5462006330490112, 0.6791588068008423, 0.6192430257797241, 0.6258729100227356, 0.6453716158866882]
```
</details>


<details>
  <summary>via the <a href="https://onnxruntime.ai/">ONNX Runtime</a></summary>

```python
# !pip install transformers onnxruntime pillow
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoTokenizer

# Load tokenizer and image processor using transformers
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(
    'jinaai/jina-clip-v2', trust_remote_code=True
)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Tokenize input texts and transform input images
input_ids = tokenizer(sentences, return_tensors='np')['input_ids']
pixel_values = image_processor(image_urls)['pixel_values']

# Start an ONNX Runtime Session
session = ort.InferenceSession('jina-clip-v2/onnx/model.onnx')

# Run inference
output = session.run(None, {'input_ids': input_ids, 'pixel_values': pixel_values})

# Keep the normalised embeddings, first 2 outputs are un-normalized
_, _, text_embeddings, image_embeddings = output
```

</details>



## License

This model is licensed to download and run under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). It is available for commercial use via the [Jina Embeddings API](https://jina.ai/embeddings/), [AWS](https://aws.amazon.com/marketplace/pp/prodview-bfbctuqmky676), [Azure](https://azuremarketplace.microsoft.com/en-gb/marketplace/apps/jinaai.jina-clip-v2-vm?tab=Overview), and [GCP](https://console.cloud.google.com/marketplace/browse?hl=en&inv=1&invt=AbiFWQ&q=jina). To download for commercial use, please [contact us](https://jina.ai/contact-sales).


## Contact

Join our [Discord community](https://discord.jina.ai) and chat with other community members about ideas.


## Citation

If you find `jina-clip-v2` useful in your research, please cite the following paper:

```bibtex
@misc{koukounas2024jinaclipv2multilingualmultimodalembeddings,
      title={jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images}, 
      author={Andreas Koukounas and Georgios Mastrapas and Bo Wang and Mohammad Kalim Akram and Sedigheh Eslami and Michael Günther and Isabelle Mohr and Saba Sturua and Scott Martens and Nan Wang and Han Xiao},
      year={2024},
      eprint={2412.08802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.08802}, 
}
```