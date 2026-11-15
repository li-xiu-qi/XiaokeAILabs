"""Inference-only shared-prefix KV branches, preserving independent causal paths.

Each branch receives a private cache. Sibling candidates never attend one
another. The trained candidate-set head runs after all branch vectors exist.
"""
import copy


def common_prefix(paths):
    # Always leave one token per candidate for extracting its final hidden state.
    limit=min(map(len, paths))-1
    for i in range(limit):
        if any(path[i]!=paths[0][i] for path in paths[1:]):return i
    return limit


def encode(engine, paths, locations, rows, mode='auto'):
    torch=engine.torch;device=engine.device;backbone=engine.model.path_encoder.backbone
    vectors=None;mask=torch.zeros(len(rows),max(len(q['candidates']) for q in rows),dtype=torch.bool,device=device)
    groups={}
    for seq,loc in zip(paths,locations):groups.setdefault(loc[0],[]).append((seq,loc))
    fallback=[];computed_tokens=0;shared_questions=0
    def save(hidden,chunk,offset=0):
        nonlocal vectors
        if vectors is None:vectors=hidden.new_zeros(mask.shape[0],mask.shape[1],hidden.shape[-1])
        for i,(seq,(qi,ci)) in enumerate(chunk):
            vectors[qi,ci]=hidden[i,len(seq)-offset-1];mask[qi,ci]=True
    for entries in groups.values():
        prefix_length=common_prefix([seq for seq,_ in entries])
        use_shared=mode=='shared' or (mode=='auto' and len(entries)>=8 and prefix_length>=128)
        if mode=='path' or not use_shared or prefix_length<1:
            fallback.extend(entries);continue
        prefix=torch.tensor([entries[0][0][:prefix_length]],device=device)
        encoded=backbone(input_ids=prefix,attention_mask=torch.ones_like(prefix),use_cache=True)
        root_cache=encoded.past_key_values;del encoded
        computed_tokens+=prefix_length;shared_questions+=1
        for start in range(0,len(entries),engine.path_batch):
            chunk=entries[start:start+engine.path_batch]
            suffixes=[seq[prefix_length:] for seq,_ in chunk];length=max(map(len,suffixes))
            ids=torch.full((len(chunk),length),engine.tokenizer.pad_token_id,dtype=torch.long,device=device)
            attention=torch.ones(len(chunk),prefix_length+length,dtype=torch.long,device=device)
            for i,suffix in enumerate(suffixes):
                ids[i,:len(suffix)]=torch.tensor(suffix,device=device)
                attention[i,prefix_length+len(suffix):]=0
            cache=copy.deepcopy(root_cache)
            cache.batch_repeat_interleave(len(chunk))
            hidden=backbone(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True).last_hidden_state
            save(hidden,chunk,prefix_length);computed_tokens+=sum(map(len,suffixes))
            del hidden,cache
        del root_cache
    for start in range(0,len(fallback),engine.path_batch):
        chunk=fallback[start:start+engine.path_batch];length=max(len(seq) for seq,_ in chunk)
        ids=torch.full((len(chunk),length),engine.tokenizer.pad_token_id,dtype=torch.long,device=device)
        attention=torch.zeros_like(ids)
        for i,(seq,_) in enumerate(chunk):
            ids[i,:len(seq)]=torch.tensor(seq,device=device);attention[i,:len(seq)]=1
        hidden=backbone(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        save(hidden,chunk);computed_tokens+=sum(len(seq) for seq,_ in chunk)
        del hidden
    return vectors,mask,{'backbone_input_tokens':computed_tokens,'shared_prefix_questions':shared_questions}
