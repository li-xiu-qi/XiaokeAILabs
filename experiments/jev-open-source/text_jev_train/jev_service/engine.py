"""Trained decision-head inference with bounded candidate microbatches."""
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from .contract import API_VERSION, prepare, encode_paths, answer


class DecisionEngine:
    def __init__(self, checkpoint, model_path, device='cuda:0', max_tokens=2048, path_batch=16, encoder='auto', temperatures=None):
        import torch
        from transformers import AutoTokenizer
        from agentjev.model import AgentJevModel
        self.torch = torch; self.device = device; self.max_tokens = max_tokens; self.path_batch = path_batch
        self.lock = threading.Lock(); self.encoder=encoder; self.temperatures={}
        if temperatures:
            calibration=json.loads(Path(temperatures).read_text())
            for kind,value in calibration.items():
                t=float(value['temperature'])
                if kind not in ('boolean','choice','score') or not math.isfinite(t) or not .05<=t<=20:
                    raise ValueError('Invalid calibration temperature')
                self.temperatures[kind]=t
        h = hashlib.sha256()
        with open(checkpoint, 'rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
        self.checkpoint_sha256 = h.hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AgentJevModel(model_path)
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if bundle.get('encoder_schema', '').startswith('routing_'):
            raise ValueError('Use a general decision checkpoint, not a routing-only fine-tune')
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.checkpoint_name = Path(checkpoint).parent.name
        del bundle
        self.model.eval().to(device)

    def info(self):
        return {'api_version': API_VERSION, 'model': 'AgentJev-0.6B', 'checkpoint': self.checkpoint_name,
                'checkpoint_sha256': self.checkpoint_sha256, 'types': ['boolean', 'choice', 'score'],
                'max_path_tokens': self.max_tokens, 'max_choice_candidates': 255,
                'output_token_decoding': False, 'shared_prefix_compute': self.encoder!='path', 'encoder':self.encoder,
                'temperatures':self.temperatures,
                'probability_semantics': 'model distribution; domain calibration is not guaranteed'}

    def evaluate(self, payload):
        prepared = prepare(payload)
        paths, locations, rows = encode_paths(prepared, self.tokenizer, self.max_tokens)
        torch = self.torch; start = time.perf_counter()
        with self.lock, torch.inference_mode(), torch.autocast(
                'cuda' if self.device.startswith('cuda') else 'cpu',
                dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            from .prefix import encode
            vectors,mask,encoding_usage=encode(self,paths,locations,rows,self.encoder)
            logits = self.model._score(vectors, mask).float().masked_fill(~mask, float('-inf'))
            if self.temperatures:
                scaling=torch.tensor([self.temperatures.get(q['type'],1.) for q in rows],device=self.device)
                logits=logits/scaling[:,None]
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        index = 0; results = []
        for request in prepared:
            answers = []
            for question in request['questions']:
                answers.append(answer(question, probabilities[index][:len(question['candidates'])])); index += 1
            results.append({'id': request['id'], 'answers': answers})
        return {'api_version': API_VERSION, 'model': self.checkpoint_name, 'results': results,
                'usage': {'questions': len(rows), 'candidate_paths': len(paths), 'input_path_tokens': sum(map(len, paths)),
                          'generated_tokens': 0, 'truncated_inputs': 0, **encoding_usage,
                          'wall_ms': round((time.perf_counter()-start)*1000, 2)}}
