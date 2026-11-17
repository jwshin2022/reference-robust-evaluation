"""Local compiled-table optimization and independently checked certificates."""
import argparse,json,hashlib,os,tempfile
from pathlib import Path
from .adaptive_search import search
from .input_contract import validate
from .table_io import score,raw
from .verify_adaptive import verify

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['solve','verify']);p.add_argument('input');p.add_argument('certificate');p.add_argument('--seconds',type=float,default=30);p.add_argument('--nodes',type=int,default=1000000);a=p.parse_args()
 f=Path(a.input);q=json.loads(f.read_text());validate(q['a'],q['b']);h=hashlib.sha256(f.read_bytes()).hexdigest();out=Path(a.certificate)
 if a.mode=='solve':
  if out.exists():p.error('Refusing to overwrite existing certificate')
  if a.seconds<=0 or a.nodes<=0:p.error('Positive limits required')
  if q['a']['domains']:
   r=search(q['a'],q['b'],seconds=a.seconds,node_cap=a.nodes)
  else:
   if min(raw(q['a'],[])[1],raw(q['b'],[])[1])<=0:raise ValueError('Positive denominators required')
   v=str(score(q['a'],q['b'],[]));r=dict(complete=True,min=v,max=v,zmin=[],zmax=[],nodes=1,terminal_prefixes=[[]],unclosed=0)
  r['input_sha256']=h
  if r['complete']:r['independent_verification']=verify(q['a'],q['b'],r)
  out.parent.mkdir(parents=True,exist_ok=True)
  # A cooperative atomic-output lock also works on filesystems without hard links.
  lock=out.with_name(out.name+'.lock');lock.mkdir()
  tmp=None
  try:
   if out.exists():raise FileExistsError(out)
   fd,tmp=tempfile.mkstemp(prefix=out.name+'.',dir=out.parent)
   data=(json.dumps(r,indent=2)+'\n').encode()
   with os.fdopen(fd,'wb')as w:w.write(data);w.flush();os.fsync(w.fileno())
   if Path(tmp).read_bytes()!=data:raise IOError('Read-back failed')
   if out.exists():raise FileExistsError(out)
   os.replace(tmp,out);tmp=None
   if out.read_bytes()!=data:raise IOError('Final read-back failed')
  finally:
   if tmp is not None and os.path.exists(tmp):os.unlink(tmp)
   lock.rmdir()
  print(json.dumps({k:r[k]for k in ['complete','min','max','nodes','input_sha256']}))
 else:
  r=json.loads(out.read_text())
  if r['input_sha256']!=h:raise ValueError('Input binding mismatch')
  print(json.dumps(verify(q['a'],q['b'],r)))
if __name__=='__main__':main()
