"""Validate the local compiled-table interchange contract before execution."""
from math import prod

def integer(value):
 if isinstance(value,bool):raise ValueError('Boolean is not an integer unit')
 if isinstance(value,int):return value
 if isinstance(value,str)and value.lstrip('-').isdigit():return int(value)
 raise ValueError('Integer unit required')

def validate(a,b):
 if a['domains']!=b['domains']:raise ValueError('Model domains differ')
 dom=a['domains']
 if any(type(k)is not int or k<1 for k in dom):raise ValueError('Positive integer domain sizes required')
 scales=[]
 for mod in [a,b]:
  c=mod['certificate'];scales.append(integer(c['area_weight_denominator']))
  for k in ['constant_numerator_units','constant_denominator_units']:integer(c[k])
  if len(mod['components'])!=len(mod['option_tables']):raise ValueError('Table count mismatch')
  used=set()
  for g,rows in zip(mod['components'],mod['option_tables']):
   vs=g['variables']
   if not vs or len(set(vs))!=len(vs)or any(type(v)is not int or not 0<=v<len(dom)for v in vs):raise ValueError('Invalid component variables')
   if used.intersection(vs):raise ValueError('Overlapping component partition')
   used.update(vs)
   if len(rows)!=prod(dom[v]for v in vs):raise ValueError('Incomplete component table')
   for r in rows:
    for k in ['numerator_units','denominator_units']:integer(r[k])
  for key in ['lower_assignment','upper_assignment']:
   z=mod[key]
   if len(z)!=len(dom)or any(type(v)is not int or not 0<=v<k for v,k in zip(z,dom)):raise ValueError('Invalid seed assignment')
 if scales[0]<=0 or scales[0]!=scales[1]:raise ValueError('Common positive area scale required')

 groups=[{i} for i in range(len(dom))]
 for mod in (a,b):
  for component in mod['components']:
   vs=set(component['variables']);hit=[g for g in groups if g&vs]
   groups=[g for g in groups if not g&vs]+[set().union(*hit)]
 if any(prod(dom[i] for i in g)>4096 for g in groups):
  raise ValueError('Joint component exceeds the public interface cap of 4096 states')
