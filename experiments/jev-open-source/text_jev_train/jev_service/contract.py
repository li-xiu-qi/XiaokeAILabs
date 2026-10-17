"""Decision API v1: transport IDs never enter semantic model input."""
import json
import math

API_VERSION = 'agentjev.decision.v1'


def semantic(value, name):
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain valid JSON') from exc
    raise ValueError(f'{name} must be nonempty text, an object or an array')


def prepare(payload):
    if not isinstance(payload, dict):
        raise ValueError('request must be an object')
    requests = payload.get('requests', [payload])
    if not isinstance(requests, list) or not 1 <= len(requests) <= 32:
        raise ValueError('requests must contain 1..32 states')
    prepared = []; total_paths = total_questions = 0
    for ri, request in enumerate(requests):
        if not isinstance(request, dict): raise ValueError('each request must be an object')
        state = semantic(request.get('state'), 'state')
        questions = request.get('questions')
        if not isinstance(questions, list) or not questions: raise ValueError('questions must be a nonempty array')
        rows = []; ids = set()
        for qi, question in enumerate(questions):
            if not isinstance(question, dict): raise ValueError('each question must be an object')
            qid = question.get('id', str(qi))
            if not isinstance(qid, str) or not qid or qid in ids:
                raise ValueError('question IDs must be unique nonempty strings within a state')
            ids.add(qid)
            kind = question.get('type', 'choice')
            text = semantic(question.get('question', question.get('instructions')), 'question')
            if kind == 'boolean':
                criteria = question.get('criteria', {})
                if not isinstance(criteria, dict): raise ValueError('criteria must be an object')
                keys = ['true', 'false']
                candidates = [semantic(criteria[k], f'{k} criterion') if k in criteria else k.upper() for k in keys]
            elif kind == 'choice':
                options = question.get('options')
                if isinstance(options, dict):
                    keys = list(options); candidates = [semantic(v, 'option') for v in options.values()]
                elif isinstance(options, list):
                    keys = [str(i) for i in range(len(options))]; candidates = [semantic(v, 'option') for v in options]
                else: raise ValueError('choice.options must be an object or array')
                if not 2 <= len(keys) <= 255: raise ValueError('choice requires 2..255 candidates')
                if any(not isinstance(k, str) or not k for k in keys): raise ValueError('option IDs must be nonempty strings')
            elif kind == 'score':
                levels = question.get('levels')
                if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                    raise ValueError('score.levels must contain 2..10 ordered descriptions')
                candidates = [semantic(v, 'level') for v in levels]; keys = [str(i) for i in range(len(levels))]
            else: raise ValueError('type must be boolean, choice or score')
            if len(set(candidates)) != len(candidates): raise ValueError('candidate descriptions must be distinct')
            rows.append({'id': qid, 'type': kind, 'text': text, 'keys': keys, 'candidates': candidates})
            total_paths += len(candidates); total_questions += 1
        prepared.append({'id': request.get('id', str(ri)), 'state': state, 'questions': rows})
    if total_questions > 128 or total_paths > 1024: raise ValueError('batch exceeds 128 questions or 1024 candidate paths')
    return prepared


def encode_paths(prepared, tokenizer, max_tokens=2048):
    paths = []; locations = []; question_rows = []
    for request in prepared:
        state = request['state']
        state_ids = tokenizer.encode(state if state.startswith('[STATE]') else '[STATE] '+state, add_special_tokens=False)
        for question in request['questions']:
            prefix = state_ids + tokenizer.encode('\n[QUESTION] '+question['text'], add_special_tokens=False)
            qindex = len(question_rows); question_rows.append(question)
            for cindex, candidate in enumerate(question['candidates']):
                ids = prefix + tokenizer.encode('\n[CANDIDATE] '+candidate, add_special_tokens=False)
                if len(ids) > max_tokens:
                    raise ValueError(f"question {question['id']!r} needs {len(ids)} tokens; limit {max_tokens}. Shorten the input; nothing was truncated.")
                paths.append(ids); locations.append((qindex, cindex))
    return paths, locations, question_rows


def answer(question, probabilities):
    if len(probabilities) != len(question['keys']) or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise RuntimeError('invalid model distribution')
    if abs(sum(probabilities)-1) > 1e-4: raise RuntimeError('model distribution does not sum to one')
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    out = {'id': question['id'], 'type': question['type'], 'distribution': dict(zip(question['keys'], probabilities))}
    if question['type'] == 'boolean': out.update(probability=probabilities[0], value=probabilities[0] >= .5)
    elif question['type'] == 'choice':
        ranked = sorted(probabilities, reverse=True)
        out.update(value=question['keys'][index], description=question['candidates'][index],
                   top_probability=ranked[0], margin=ranked[0]-ranked[1])
    else: out.update(score=sum(i*p for i, p in enumerate(probabilities)), level=index, legend=question['candidates'])
    return out
