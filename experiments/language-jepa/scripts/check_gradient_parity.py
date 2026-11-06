import argparse, copy, importlib.util, torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments
parser=argparse.ArgumentParser(); parser.add_argument('--trainer',required=True); opts=parser.parse_args()
spec=importlib.util.spec_from_file_location('author',opts.trainer)
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
torch.manual_seed(42);torch.set_num_threads(1)
c=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,attention_dropout=0)
m=Qwen2ForCausalLM(c); n=copy.deepcopy(m)
args=TrainingArguments(output_dir='/tmp/jepa-gradient-probe',use_cpu=True,report_to='none',gradient_accumulation_steps=4)
t1=Trainer(model=m,args=args); t2=a.RepresentationTrainer(model=n,args=args,lbd=0.0,gamma=1.0,last_token=-1)
for t in [t1,t2]: t.model_accepts_loss_kwargs=False; t.current_gradient_accumulation_steps=4
ids=torch.tensor([[1,2,3,4,5,6],[1,4,3,6,5,2]]);labels=torch.tensor([[-100,-100,3,4,-100,-100],[-100,-100,3,6,-100,-100]]); mask=torch.ones_like(ids)
b={'input_ids':ids,'attention_mask':mask,'labels':labels}
j=dict(b,input_ids_user=ids,attention_mask_user=mask,labels_user=torch.full_like(ids,-100),input_ids_assistant=ids,attention_mask_assistant=mask,labels_assistant=torch.full_like(ids,-100))
l1=t1.training_step(m,b,num_items_in_batch=torch.tensor(16)); l2=t2.training_step(n,j,num_items_in_batch=torch.tensor(16))
errors=[(x.grad-y.grad).abs().max().item() for x,y in zip(m.parameters(),n.parameters()) if x.grad is not None and y.grad is not None]
assert len(errors)>0 and max(errors)<1e-6, max(errors)
print({'baseline_loss':l1.item(),'jepa_lambda0_loss':l2.item(),'max_gradient_difference':max(errors),'parameters_compared':len(errors),'accumulation_steps':4})
