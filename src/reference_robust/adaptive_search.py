"""Exact incumbent residual support tests for a compiled paired objective.
Candidate proposals are heuristics; only complete checked tree coverage certifies extrema.
No float is used to order or prune objective values.
"""
from fractions import Fraction as F
from itertools import product
import time, math
from .support_hull import upper_hull,query
from .table_io import tables,raw,score

def decode(tabs, choices, domains):
    z=[0]*len(domains)
    for (vs,rows),idx in zip(tabs,choices):
        dims=[domains[v] for v in vs]
        vals=[]
        for k in reversed(dims):vals.append(idx%k);idx//=k
        for v,a in zip(vs,reversed(vals)):z[v]=a
    return z

def coefficients(t,ar,br):
    an,ad=ar.numerator,ar.denominator;bn,bd=br.numerator,br.denominator
    return bd*(ad*t[0]-an*t[1]),ad*(bd*t[2]-bn*t[3])

def support_certificate(base,tabs,pre,ar,br,side):
    """Standalone exact support test; used only for qualification, not timed search."""
    t=list(base)
    for i,j in enumerate(pre):t=[a+b for a,b in zip(t,tabs[i][1][j])]
    rest=tabs[len(pre):]
    dl=t[1]+sum(min(row[1]for row in rows)for _,rows in rest)
    du=t[1]+sum(max(row[1]for row in rows)for _,rows in rest)
    el=t[3]+sum(min(row[3]for row in rows)for _,rows in rest)
    eu=t[3]+sum(max(row[3]for row in rows)for _,rows in rest)
    if min(dl,el)<=0:raise ValueError('Positive denominators required')
    op=min if side=='min' else max
    aa,bb=coefficients(t,ar,br);bounds=[]
    for rn,rd in [(dl,el),(du,eu)]:
        v=rd*aa-rn*bb
        for _,rows in rest:
            v+=op(rd*c-rn*d for c,d in [coefficients(row,ar,br)for row in rows])
        bounds.append(v)
    return min(bounds)>=0 if side=='min' else max(bounds)<=0

def search(a,b,seconds=30,node_cap=1000000,proposal=True):
    t0=time.monotonic();base,tabs,K=tables(a,b);n=len(tabs);domains=a['domains']
    assert domains==b['domains']
    for _,rows in tabs:assert len({r[3]-r[1]for r in rows})==1
    lows=[[0]*4 for _ in range(n+1)];highs=[[0]*4 for _ in range(n+1)]
    for i in reversed(range(n)):
        for h in range(4):
            lows[i][h]=lows[i+1][h]+min(t[h]for t in tabs[i][1])
            highs[i][h]=highs[i+1][h]+max(t[h]for t in tabs[i][1])
    if min(base[1]+lows[0][1],base[3]+lows[0][3])<=0:raise ValueError('Positive denominator bound required')
    seeds=list(dict.fromkeys(tuple(z)for z in [[0]*len(domains),[k-1 for k in domains]]+[r[s+'_assignment']for r in [a,b]for s in ['lower','upper']]))
    values=[(score(a,b,z),list(z))for z in seeds]
    lo,zlo=min(values);hi,zhi=max(values);proposals=0
    if proposal:
        for side in ['min','max']:
            best=lo if side=='min'else hi;z=zlo[:]if side=='min'else zhi[:]
            for _ in range(16):
                na,da=raw(a,z);nb,db=raw(b,z);ar,br=F(na,da),F(nb,db)
                chosen=[]
                for vs,rows in tabs:
                    vals=[db*c-da*d for c,d in [coefficients(r,ar,br)for r in rows]]
                    chosen.append((min if side=='min'else max)(range(len(vals)),key=lambda j:vals[j]))
                zz=decode(tabs,chosen,domains);v=score(a,b,zz);proposals+=1
                if (v<best if side=='min'else v>best):best,z=v,zz
                else:break
            if side=='min':lo,zlo=best,z
            else:hi,zhi=best,z
    prep=time.monotonic()-t0
    cache={}
    def test(t,i,z,side):
        key=(side,tuple(z))
        if key not in cache:
            ar=F(*raw(a,z));br=F(*raw(b,z))
            rows=[ [coefficients(r,ar,br)for r in rs]for _,rs in tabs]
            cache[key]=[ar,br,rows,None]
        ar,br,rows,hulls=cache[key];aa,bb=coefficients(t,ar,br)
        dl,du=t[1]+lows[i][1],t[1]+highs[i][1];el,eu=t[3]+lows[i][3],t[3]+highs[i][3]
        sg=-1 if side=='min'else 1
        # The root uses direct support. Build envelopes only when branching makes reuse possible.
        if i>0 and hulls is None:
            hulls=[upper_hull([(-sg*y,sg*x)for x,y in rr])for rr in rows]
            cache[key][3]=hulls
        for rn,rd in [(dl,el),(du,eu)]:
            if hulls is None:
                op=min if side=='min'else max
                v=rd*aa-rn*bb+sum(op(rd*x-rn*y for x,y in rr)for rr in rows[i:])
            else:v=rd*aa-rn*bb+sg*sum(query(hulls[j],rn,rd)for j in range(i,n))
            if(v<0 if side=='min'else v>0):return False
        return True
    stack=[((),base)];terminal=[];nodes=0
    while stack and nodes<node_cap and time.monotonic()-t0<seconds:
        pre,t=stack.pop();i=len(pre);nodes+=1
        if i==n:
            v=F(t[0],t[1])-F(t[2],t[3]);z=decode(tabs,pre,domains)
            if v<lo:lo,zlo=v,z
            if v>hi:hi,zhi=v,z
            terminal.append(list(pre));continue
        if test(t,i,zlo,'min')and test(t,i,zhi,'max'):
            terminal.append(list(pre));continue
        for j in reversed(range(len(tabs[i][1]))):
            stack.append((pre+(j,),[u+v for u,v in zip(t,tabs[i][1][j])]))
    return {'complete':not stack,'min':str(lo),'max':str(hi),'zmin':zlo,'zmax':zhi,'nodes':nodes,
            'terminal_prefixes':terminal,'unclosed':len(stack),'seconds_solve':time.monotonic()-t0,
            'seconds_preparation':prep,'proposal_steps':proposals,'components':n,'max_options':max(len(r)for _,r in tabs),
            'method':'adaptive_residual','certificate':'exact per-component support at two common-area ratio endpoints',
            'stop_reason':None if not stack else 'node_cap'if nodes>=node_cap else'time_limit'}
