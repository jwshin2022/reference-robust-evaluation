from fractions import Fraction as F
import itertools,math

def raw(r,z):
 n=int(r['certificate']['constant_numerator_units']);d=int(r['certificate']['constant_denominator_units'])
 for g,tab in zip(r['components'],r['option_tables']):
  index=0;mul=1
  for v in g['variables']:index+=z[v]*mul;mul*=r['domains'][v]
  n+=int(tab[index]['numerator_units']);d+=int(tab[index]['denominator_units'])
 return n,d

def score(a,b,z):
 na,da=raw(a,z);nb,db=raw(b,z);return F(na,da)-F(nb,db)

def tables(a,b):
 dom=a['domains'];groups=[{v}for v in range(len(dom))]
 for mod in [a,b]:
  for group in mod['components']:
   hit=[g for g in groups if g&set(group['variables'])];groups=[g for g in groups if g not in hit]+[set().union(*hit)]
 base=[int(mod['certificate'][k])for mod in [a,b]for k in ['constant_numerator_units','constant_denominator_units']];out=[]
 for group in groups:
  vs=sorted(group);rows=[]
  for z0 in itertools.product(*(range(dom[v])for v in vs)):
   z=dict(zip(vs,z0));row=[]
   for mod in [a,b]:
    n=d=0
    for g,tab in zip(mod['components'],mod['option_tables']):
     if set(g['variables'])<=group:
      ind=0;mul=1
      for v in g['variables']:ind+=z[v]*mul;mul*=dom[v]
      n+=int(tab[ind]['numerator_units']);d+=int(tab[ind]['denominator_units'])
    row.extend([n,d])
   rows.append(row)
  assert len({r[3]-r[1]for r in rows})==1;out.append((vs,rows))
 out.sort(key=lambda x:max(r[0]-r[2]for r in x[1])-min(r[0]-r[2]for r in x[1]),reverse=True)
 K=base[3]-base[1]+sum(rows[0][3]-rows[0][1]for _,rows in out)
 return base,out,K

def bound(base,tabs,K,pre,coupled):
 t=list(base)
 for i,idx in enumerate(pre):t=[a+b for a,b in zip(t,tabs[i][1][idx])]
 rng=[(t[h]+sum(min(row[h]for row in rows)for _,rows in tabs[len(pre):]),t[h]+sum(max(row[h]for row in rows)for _,rows in tabs[len(pre):]))for h in range(4)]
 (al,au),(dl,du),(bl,bu),(el,eu)=rng
 l,u=F(al,du)-F(bu,el),F(au,dl)-F(bl,eu)
 if coupled:
  nl=t[0]-t[2]+sum(min(row[0]-row[2]for row in rows)for _,rows in tabs[len(pre):]);nu=t[0]-t[2]+sum(max(row[0]-row[2]for row in rows)for _,rows in tabs[len(pre):])
  corners=[(n+K*q)/d for n,q,d in itertools.product([nl,nu],[F(bl,eu),F(bu,el)],[dl,du])];l=max(l,min(corners));u=min(u,max(corners))
 return l,u
