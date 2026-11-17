"""Independent rational-breakpoint verifier of exact residual tree certificates.
Same exact criterion as verify_residual.py; no optimizer/search import.
"""
from fractions import Fraction as F
from math import prod
from functools import lru_cache
from bisect import bisect_right
from .table_io import tables,raw,score

def verify(a,b,result):
    if not result['complete']:raise ValueError('Incomplete endpoint certificate')
    domains=a['domains']
    for z in [result['zmin'],result['zmax']]:
        if len(z)!=len(domains)or any(type(v)is not int or not 0<=v<k for v,k in zip(z,domains)):raise ValueError('Invalid witness')
    if str(score(a,b,result['zmin']))!=result['min']or str(score(a,b,result['zmax']))!=result['max']:raise ValueError('Wrong score')
    base,tabs,K=tables(a,b);sizes=[len(rows)for _,rows in tabs];n=len(tabs)
    low=[[0]*4 for _ in range(n+1)];high=[[0]*4 for _ in range(n+1)]
    for i in reversed(range(n)):
        for h in range(4):low[i][h]=low[i+1][h]+min(row[h]for row in tabs[i][1]);high[i][h]=high[i+1][h]+max(row[h]for row in tabs[i][1])
    pars=[];coefs=[]
    for side in ['min','max']:
        z=result['z'+side];alpha=F(*raw(a,z));beta=F(*raw(b,z));pars.append((alpha,beta))
        # Compute a rational residual once, then clear its common positive scale.
        scale=alpha.denominator*beta.denominator
        coefs.append([[(int((F(r[0])-alpha*r[1])*scale),int((F(r[2])-beta*r[3])*scale))for r in rows]for _,rows in tabs])
    # Independently construct rational breakpoints from the verifier's residuals.
    # Does not import the search or its hull implementation.
    direct=result['terminal_prefixes']==[[]]
    envelopes=[]
    for side in range(0 if direct else 2):
        sg=-1 if side==0 else 1;group=[]
        for rows in coefs[side]:
            dedup={}
            for x,y in rows:
                slope=-sg*y;intercept=sg*x
                dedup[slope]=max(dedup.get(slope,intercept),intercept)
            lines=[];starts=[]
            for slope,intercept in sorted(dedup.items()):
                while lines:
                    prev_s,prev_b=lines[-1]
                    cross=F(prev_b-intercept,slope-prev_s)
                    if len(lines)==1 or cross>starts[-1]:break
                    lines.pop();starts.pop()
                starts.append(None if not lines else cross);lines.append((slope,intercept))
            group.append((lines,[(v.numerator,v.denominator)if v is not None else None for v in starts]))
        envelopes.append(group)
    @lru_cache(maxsize=16384)
    def suffix(side,i,rn,rd):
        if direct:
            op=min if side==0 else max
            return sum(op(rd*x-rn*y for x,y in rr)for rr in coefs[side][i:])
        total=0
        for lines,starts in envelopes[side][i:]:
            l,h=1,len(starts)
            while l<h:
                mid=(l+h)//2;p,q=starts[mid]
                if p*rd<=rn*q:l=mid+1
                else:h=mid
            slope,intercept=lines[l-1];total+=rn*slope+rd*intercept
        return (-1 if side==0 else 1)*total
    covered=0;trie={}
    for prefix in result['terminal_prefixes']:
        if len(prefix)>n:raise ValueError('Prefix length')
        node=trie;t=list(base)
        for i,j in enumerate(prefix):
            if type(j)is not int or not 0<=j<sizes[i]:raise ValueError('Prefix value')
            if 'end'in node:raise ValueError('Overlapping prefix')
            node=node.setdefault(j,{});t=[x+y for x,y in zip(t,tabs[i][1][j])]
        if node:raise ValueError('Duplicate or overlapping prefix')
        node['end']=True;depth=len(prefix);covered+=prod(sizes[depth:])
        dl,du=t[1]+low[depth][1],t[1]+high[depth][1];el,eu=t[3]+low[depth][3],t[3]+high[depth][3]
        if min(dl,el)<=0 or el-dl!=K or eu-du!=K:raise ValueError('Denominator identity/domain')
        for side,(alpha,beta)in enumerate(pars):
            scale=alpha.denominator*beta.denominator
            x=int((F(t[0])-alpha*t[1])*scale);y=int((F(t[2])-beta*t[3])*scale)
            for rn,rd in [(dl,el),(du,eu)]:
                val=rd*x-rn*y+suffix(side,depth,rn,rd)
                if(val<0 if side==0 else val>0):raise ValueError('Residual support fails')
    if covered!=prod(sizes):raise ValueError('Incomplete tree cover')
    return {'verified':True,'covered_joint_choices':str(covered),'terminal_prefixes':len(result['terminal_prefixes']),'suffix_cache':str(suffix.cache_info())}
