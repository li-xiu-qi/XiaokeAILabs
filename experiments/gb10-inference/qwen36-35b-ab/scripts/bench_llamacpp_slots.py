import json, time, os, sys, subprocess, urllib.request, concurrent.futures, threading, statistics, traceback, socket
from pathlib import Path
D=Path(os.environ.get('BENCH_OUT', './bench-output')); D.mkdir(parents=True,exist_ok=False)
BIN=os.environ['LLAMA_SERVER']
MODEL=os.environ['MODEL_GGUF']
PORT=19433
prompts=['Explain how a database index speeds up queries.','Describe three practical ways to evaluate a retrieval system.','Explain the tradeoffs between latency and throughput in model inference.','Describe how to debug an unexpectedly slow Python service.','Explain why training and test datasets should be separated.','Describe how caching improves a web application.','Explain how batching changes GPU inference throughput.','Describe practical checks for a machine learning dataset.']
stop=threading.Event()
monitor_ready=threading.Event()
monitor_failed=threading.Event()
monitor_thread=None
server_lock=threading.Lock()
current_server=None

def emergency_stop():
 global current_server
 monitor_failed.set()
 stop.set()
 with server_lock:
  if current_server is not None and current_server.poll() is None:
   try: current_server.kill()
   except ProcessLookupError: pass

def monitor():
 try:
  with open(D/'resources.jsonl','a') as f:
   while not stop.is_set():
    m={x.split(':')[0]:int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith(('MemAvailable:','SwapFree:'))}
    if 'MemAvailable' not in m or 'SwapFree' not in m: raise ValueError('missing required memory fields')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
    gpu_lines = gpu.stdout.strip().splitlines()
    if not gpu_lines or any(len(fields := line.split(',')) != 3 or any(not x.strip() for x in fields) for line in gpu_lines):
     raise ValueError('nvidia-smi returned incomplete output')
    f.write(json.dumps({'time':time.time(),'mem_kb':m,'nvidia':gpu.stdout.strip()})+'\n'); f.flush()
    if m['MemAvailable'] < 18*1024*1024: emergency_stop(); return
    monitor_ready.set()
    stop.wait(2)
 except Exception as exc:
  print(f'resource monitor failed: {exc!r}', file=sys.stderr, flush=True)
  emergency_stop()
def request(i, label):
 data=json.dumps({'prompt':prompts[i%len(prompts)],'n_predict':128,'temperature':0,'seed':100+i,'ignore_eos':True,'cache_prompt':False}).encode()
 t=time.monotonic()
 try:
  req=urllib.request.Request('http://127.0.0.1:%d/completion'%PORT,data=data,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=240) as resp: obj=json.load(resp)
  return {'id':i,'label':label,'elapsed_s':time.monotonic()-t,'tokens_predicted':obj.get('tokens_predicted'),'tokens_evaluated':obj.get('tokens_evaluated'),'timings':obj.get('timings'),'stop':obj.get('stopped_limit'),'error':obj.get('error')}
 except Exception as e: return {'id':i,'label':label,'elapsed_s':time.monotonic()-t,'error':repr(e)}
def validate_round(rows, interrupted):
 if interrupted: raise RuntimeError('resource monitor interrupted round')
 if len(rows) != 8 or any(x.get('error') or x.get('tokens_predicted') != 128 for x in rows):
  raise RuntimeError('request failed or incomplete round')

def finish_monitor(thread):
 stop.set()
 thread.join(timeout=6)
 if thread.is_alive(): raise RuntimeError("resource monitor did not stop")
 if monitor_failed.is_set(): raise RuntimeError("resource monitor failed before completion")

def main():
 global current_server, monitor_thread
 probe=socket.socket(); probe.settimeout(1)
 try:
  if probe.connect_ex(('127.0.0.1', PORT)) == 0: raise RuntimeError('benchmark port already in use')
 finally: probe.close()
 print('start',flush=True); monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()
 if not monitor_ready.wait(10) or stop.is_set(): raise RuntimeError('resource monitor not ready or memory guard triggered')
 allres=[]
 for slots in (1,8):
  if stop.is_set(): raise RuntimeError('memory guard triggered')
  cmd=[BIN,'-m',MODEL,'-ngl','99','-c',str(4096*slots),'-np',str(slots),'-t','8','--cache-type-k','q8_0','--cache-type-v','q8_0','--host','127.0.0.1','--port',str(PORT),'--jinja','--spec-type','draft-mtp','--spec-draft-n-max','2','--spec-draft-backend-sampling']
  (D/('command-%s.json'%slots)).write_text(json.dumps(cmd,indent=2))
  with open(D/('server-%s.log'%slots),'w') as log:
   with server_lock:
    if stop.is_set(): raise RuntimeError('memory guard triggered')
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    current_server=p
   try:
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
     if stop.is_set(): raise RuntimeError('memory guard triggered')
     if p.poll() is not None: raise RuntimeError('server exited '+str(p.returncode))
     try:
      with urllib.request.urlopen('http://127.0.0.1:%d/health'%PORT,timeout=2) as r:
       if r.status==200: break
     except Exception: time.sleep(2)
    else: raise TimeoutError('server startup')
    smoke=request(0,'smoke'); print('slots',slots,'smoke',smoke,flush=True)
    if smoke.get('error') or smoke.get('tokens_predicted')!=128: raise RuntimeError('smoke failed')
    for concurrency in (1,8):
     for run in (1,2):
      if stop.is_set(): raise RuntimeError('memory guard triggered')
      start=time.monotonic()
      with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
       rows=list(ex.map(lambda i: request(i,'measured'),range(8)))
      wall=time.monotonic()-start
      validate_round(rows, stop.is_set())
      row={'slots':slots,'concurrency':concurrency,'run':run,'wall_s':wall,'requests':rows,'tokens':sum(x.get('tokens_predicted',0) or 0 for x in rows)}
      allres.append(row); (D/'results.json').write_text(json.dumps(allres,ensure_ascii=False,indent=2)); print(json.dumps({'slots':slots,'concurrency':concurrency,'run':run,'wall':wall,'tokens':row['tokens'],'errors':[x.get('error') for x in rows if x.get('error')]}),flush=True)
   finally:
    with server_lock:
     if p.poll() is None: p.terminate()
     current_server=None
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    time.sleep(5)
 if stop.is_set(): raise RuntimeError('resource monitor failed before completion')
 finish_monitor(monitor_thread)
 print('complete',flush=True)
(D/'run-status.json').write_text(json.dumps({'status':'running'}))
try:
 main()
 (D/'run-status.json').write_text(json.dumps({'status':'complete'}))
except Exception as exc:
 stop.set()
 (D/'run-status.json').write_text(json.dumps({'status':'failed','error':repr(exc)}))
 traceback.print_exc(); sys.exit(1)
