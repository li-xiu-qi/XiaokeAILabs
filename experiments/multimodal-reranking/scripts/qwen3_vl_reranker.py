# Adapted from Qwen3-VL-Embedding, Apache-2.0:
# https://github.com/QwenLM/Qwen3-VL-Embedding
# Modified for Transformers 5.x token-type tensor compatibility.
# Upstream reference at audit: https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_reranker.py
import os
import torch
import numpy as np
import logging
import unicodedata

from PIL import Image
from scipy import special
from typing import List, Union, Optional, Dict
from urllib.parse import urlparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

# Default configuration constants
MAX_LENGTH = 10240
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 tokens
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR  # 1800 tokens
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS  # 7680 tokens


def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path

    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True

    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]

        if isinstance(first_elem, Image.Image):
            return True

        if isinstance(first_elem, str):
            return is_image_path(first_elem)

    return False


def sample_frames(
    frames: List[Union[str, Image.Image]],
    max_segments: int
) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [frames[frame_idx] for frame_idx in frame_id_list]
    return sampled_frames


class Qwen3VLReranker():

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        # Load the language model
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs
        ).to(self.device)

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left'
        )
        self.model.eval()

        # Initialize binary classification head for yes/no scoring
        token_true_id = self.processor.tokenizer.get_vocab()["yes"]
        token_false_id = self.processor.tokenizer.get_vocab()["no"]
        self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)

    def get_binary_linear(self, model, token_yes: int, token_no: int) -> torch.nn.Linear:
        lm_head_weights = model.lm_head.weight.data

        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]

        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs: Dict) -> List[float]:
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def truncate_tokens_optimized(
        self,
        tokens: List[str],
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)

        # Calculate budget: how many non-special tokens we can keep
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special

        # Build final list according to budget
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1

        return final_tokens

    def tokenize(self, pairs: List[Dict], **kwargs) -> Dict:
        max_length = self.max_length
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)

        try:
            images, videos, video_kwargs = process_vision_info(
                pairs,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True,
                tokenize=False
            )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )

        # Truncate input IDs while preserving special tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5],
                max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]

        # Apply padding
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]

        # Transformers 5.x expects multimodal token types as a tensor. Since the
        # processor produced them before manual truncation, reject mismatched
        # lengths instead of silently scoring misaligned visual tokens.
        if "mm_token_type_ids" in inputs:
            types = inputs["mm_token_type_ids"]
            if not isinstance(types, torch.Tensor):
                types = torch.as_tensor(types, dtype=torch.long)
            if types.shape != inputs["input_ids"].shape:
                raise ValueError("multimodal token types do not align with padded/truncated input IDs")
            inputs["mm_token_type_ids"] = types

        return inputs

    def format_mm_content(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        prefix: str = 'Query:',
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        content = []
        content.append({'type': 'text', 'text': prefix})

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}

            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video',
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None

            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image',
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        return content

    def format_mm_instruction(
        self,
        query_text: Optional[Union[str, tuple]] = None,
        query_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        query_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        doc_text: Optional[Union[List[str], str]] = None,
        doc_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        doc_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        })

        # Handle query_text as tuple containing (instruction, text)
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction

        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + (instruct or self.default_instruction)
        })

        # Format query content
        query_content = self.format_mm_content(
            query_text, query_image, query_video,
            prefix='<Query>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(query_content)

        # Format document content
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video,
            prefix='\n<Document>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(doc_content)

        inputs.append({
            "role": "user",
            "content": contents
        })

        return inputs

    def process(
        self,
        inputs: Dict,
    ) -> List[float]:
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        if not query or not documents:
            return []

        # Format each query-document pair
        pairs = [
            self.format_mm_instruction(
                query.get('text', None),
                query.get('image', None),
                query.get('video', None),
                document.get('text', None),
                document.get('image', None),
                document.get('video', None),
                instruction=instruction,
                fps=inputs.get('fps', self.fps),
                max_frames=inputs.get('max_frames', self.max_frames)
            )
            for document in documents
        ]

        # Compute scores for each pair
        final_scores = []
        for pair in pairs:
            tokenized_inputs = self.tokenize([pair])
            tokenized_inputs = tokenized_inputs.to(self.model.device)
            scores = self.compute_scores(tokenized_inputs)
            final_scores.extend(scores)

        return final_scores