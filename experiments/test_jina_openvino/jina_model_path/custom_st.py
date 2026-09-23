import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


class Transformer(nn.Module):

    save_in_root: bool = True
    
    def __init__(
        self,
        model_name_or_path: str = 'jinaai/jina-clip-v2',
        tokenizer_name_or_path: Optional[str] = None,
        image_processor_name_or_path: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        config_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        image_processor_args: Optional[Dict[str, Any]] = None,
        assume_text_inputs: bool = False,
        cache_dir: Optional[str] = None,
        backend: Literal['torch', 'onnx', 'openvino'] = 'torch',
        **_,
    ) -> None:
        """
        Creates a custom SentenceTransformer module that uses `jinai/jina-clip-v2` to
        map sentences/images to embeddings

        Args:
            model_name_or_path (str, optional): If it is a filepath on disc, it loads
                the model from that path. If it is not a path, tries to construct a
                model from the Hugging Face Hub with that name. Defaults to
                'jinaai/jina-clip-v2'
            tokenizer_name_or_path (str, optional): If it is a filepath on disc, it
                loads the tokenizer from that path. If it is not a path, tries to
                construct a tokenizer from the Hugging Face Hub with that name.
                If `None` it is automatically set to the value of `model_name_or_path`
            image_processor_name_or_path (str, optional): If it is a filepath on disc,
                it loads the image processor from that path. If it is not a path, tries
                to construct an image processor from the Hugging Face Hub with that
                name. If `None` it is automatically set to the value of
                `model_name_or_path`
            max_seq_length (int, optional): The maximum sequence length of the model.
                If not provided, will be inferred from model or tokenizer
            config_args (Dict[str, Any], optional): Additional model configuration
                parameters to be passed to the Hugging Face Transformers config
            model_args (Dict[str, Any], optional): Additional model configuration
                parameters to be passed to the Hugging Face Transformers model
            tokenizer_args (Dict[str, Any], optional): Additional tokenizer
                configuration parameters to be passed to the Hugging Face Transformers
                tokenizer
            image_processor_args (Dict[str, Any], optional): Additional image processor
                configuration parameters to be passed to the Hugging Face Transformers
                image processor
            assume_text_inputs (bool, optional): If set to `True`, all inputs are
                treated as texts. Defaults to `False`
            cache_dir (str, optional): The Hugging Face Hub cache directory
            backend (str, optional): Computational backend, only 'torch' is supported

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    'jinaai/jina-clip-v2', trust_remote_code=True
                )
                sentences_or_images = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "/path/to/stadium.jpg",
                ]
                embeddings = model.encode(sentences_or_images)
                print(embeddings.shape)
                # (3, 1024)

                # Get the similarity scores between all inputs
                similarities = model.similarity(embeddings, embeddings)
                print(similarities)
                # tensor([[1.0000, 0.6817, 0.0492],
                #         [0.6817, 1.0000, 0.0421],
                #         [0.0492, 0.0421, 1.0000]])
        """
        super(Transformer, self).__init__()
        if backend != 'torch':
            raise ValueError(
                f'Backend \'{backend}\' is not supported, please use \'torch\' instead'
            )

        config_kwargs = config_args or {}
        model_kwargs = model_args or {}
        tokenizer_kwargs = tokenizer_args or {}
        image_processor_kwargs = {
            'token': model_kwargs.get('token', None),
            'trust_remote_code': model_kwargs.get('trust_remote_code', False),
            'revision': model_kwargs.get('revision', None),
            'local_files_only': model_kwargs.get('local_files_only', None),
        }
        image_processor_kwargs.update(image_processor_args or {})

        config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **config_kwargs
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_kwargs
        )
        if max_seq_length is not None and 'model_max_length' not in tokenizer_kwargs:
            tokenizer_kwargs['model_max_length'] = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_name_or_path or model_name_or_path,
            cache_dir=cache_dir,
            **image_processor_kwargs,
        )
        self.assume_text_inputs = assume_text_inputs

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.model, 'config')
                and hasattr(self.model.config, 'max_position_embeddings')
                and hasattr(self.tokenizer, 'model_max_length')
            ):
                max_seq_length = min(
                    self.model.config.max_position_embeddings,
                    self.tokenizer.model_max_length,
                )
        self.max_seq_length = max_seq_length
        if tokenizer_name_or_path is not None:
            self.model.config.tokenizer_class = self.tokenizer.__class__.__name__

    @staticmethod
    def _decode_data_image(data_image_str: str) -> Image.Image:
        header, data = data_image_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

    def tokenize(
        self, texts: List[Union[str, Image.Image]], padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes input samples. Text samples are tokenized. Image URLs, image data
        buffers and PIL images are passed through the image processor.
        """
        _images = []
        _texts = []
        _image_or_text_descriptors = []

        if self.assume_text_inputs:
            for sample in texts:
                if isinstance(sample, str):
                    _texts.append(sample)
                    _image_or_text_descriptors.append(1)
        else:
            for sample in texts:
                if isinstance(sample, str):
                    if sample.startswith('http'):
                        try:
                            response = requests.get(sample)
                            _images.append(
                                Image.open(BytesIO(response.content)).convert('RGB')
                            )
                            _image_or_text_descriptors.append(0)
                        except Exception as e:
                            _ = str(e)
                            _texts.append(sample)
                            _image_or_text_descriptors.append(1)
                    elif sample.startswith('data:image/'):
                        _images.append(self._decode_data_image(sample).convert('RGB'))
                        _image_or_text_descriptors.append(0)
                    else:
                        try:
                            _images.append(Image.open(sample).convert('RGB'))
                            _image_or_text_descriptors.append(0)
                        except Exception as e:
                            _ = str(e)
                            _texts.append(sample)
                            _image_or_text_descriptors.append(1)
                elif isinstance(sample, Image.Image):
                    _images.append(sample.convert('RGB'))
                    _image_or_text_descriptors.append(0)

        encoding = {}
        if len(_texts):
            encoding['input_ids'] = self.tokenizer(
                _texts,
                padding=padding,
                truncation='longest_first',
                return_tensors='pt',
                max_length=self.max_seq_length,
            ).input_ids

        if len(_images):
            encoding['pixel_values'] = self.image_processor(
                _images, return_tensors='pt'
            ).pixel_values

        encoding['image_text_info'] = _image_or_text_descriptors
        return encoding

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_embeddings = []
        text_embeddings = []

        if 'pixel_values' in features:
            image_embeddings = self.model.get_image_features(features['pixel_values'])
        if 'input_ids' in features:
            text_embeddings = self.model.get_text_features(features['input_ids'])

        sentence_embedding = []
        image_features = iter(image_embeddings)
        text_features = iter(text_embeddings)
        for _, _input_type in enumerate(features['image_text_info']):
            if _input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features['sentence_embedding'] = torch.stack(sentence_embedding).float()
        return features

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)
        self.image_processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str) -> 'Transformer':
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            'sentence_bert_config.json',
            'sentence_roberta_config.json',
            'sentence_distilbert_config.json',
            'sentence_camembert_config.json',
            'sentence_albert_config.json',
            'sentence_xlm-roberta_config.json',
            'sentence_xlnet_config.json',
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)

        # Don't allow configs to set trust_remote_code
        if 'config_kwargs' in config and 'trust_remote_code' in config['config_kwargs']:
            config['config_kwargs'].pop('trust_remote_code')
        if 'model_kwargs' in config and 'trust_remote_code' in config['model_kwargs']:
            config['model_kwargs'].pop('trust_remote_code')
        if (
            'tokenizer_kwargs' in config
            and 'trust_remote_code' in config['tokenizer_kwargs']
        ):
            config['tokenizer_kwargs'].pop('trust_remote_code')
        if (
            'image_processor_kwargs' in config
            and 'trust_remote_code' in config['image_processor_kwargs']
        ):
            config['image_processor_kwargs'].pop('trust_remote_code')

        return Transformer(model_name_or_path=input_path, **config)
