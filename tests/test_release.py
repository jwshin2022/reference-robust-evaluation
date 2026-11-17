import unittest,json,itertools,copy
from pathlib import Path
from fractions import Fraction
from reference_robust.cli import evaluate
from reference_robust.input_contract import validate
from reference_robust.adaptive_search import search
from reference_robust.verify_adaptive import verify
from reference_robust.table_io import score
R=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 def test_records(self):
  r=evaluate([dict(case='x',reference='a',score_a='9/10',score_b='4/5'),dict(case='x',reference='b',score_a='3/4',score_b='9/10')],Fraction(1,100))
  self.assertEqual(r[0]['decision'],'REVERSAL')
 def test_margin_tie(self):
  r=evaluate([dict(case='x',reference='a',score_a='81/100',score_b='4/5')],Fraction(1,100))
  self.assertEqual(r[0]['decision'],'UNRESOLVED')
 def test_duplicate(self):
  row=dict(case='x',reference='a',score_a='1',score_b='0')
  with self.assertRaises(ValueError):evaluate([row,row],Fraction(0))
 def test_compiled_exact_and_tamper(self):
  q=json.loads((R/'examples/compiled_pair.json').read_text());a,b=q['a'],q['b'];validate(a,b)
  r=search(a,b,seconds=30);self.assertTrue(r['complete']);self.assertTrue(verify(a,b,r)['verified'])
  vals=[score(a,b,z)for z in itertools.product(*(range(k)for k in a['domains']))]
  self.assertEqual(Fraction(r['min']),min(vals));self.assertEqual(Fraction(r['max']),max(vals))
  for key,value in [('min','17'),('terminal_prefixes',[]),('zmax',[-1]*len(a['domains']))]:
   bad=copy.deepcopy(r);bad[key]=value
   with self.assertRaises((ValueError,AssertionError,IndexError)):verify(a,b,bad)
 def test_scale_refusal(self):
  q=json.loads((R/'examples/compiled_pair.json').read_text());q['b']['certificate']['area_weight_denominator']='999'
  with self.assertRaises(ValueError):validate(q['a'],q['b'])
 def test_geometry(self):
  try:import numpy as np
  except ImportError:self.skipTest('optional geometry dependencies unavailable')
  from reference_robust.geometry import compile_pair
  from reference_robust.completion_envelope import _codes,_weights,_exact_footprint
  from reference_robust.vendor.surface_distance import lookup_tables
  from scipy import ndimage
  # Full-union oracle enumerates every reference without component factorization.
  def direct(P,T,spacing,tau):
   pc=_codes(P);tc=_codes(T);w=[Fraction(float(x))for x in lookup_tables.create_table_neighbour_code_to_surface_area(spacing)]
   p=(pc!=0)&(pc!=255);t=(tc!=0)&(tc!=255)
   den=sum((w[int(x)]for x in pc[p]),Fraction(0))+sum((w[int(x)]for x in tc[t]),Fraction(0))
   from reference_robust.geometry import boundary_codes
   self.assertTrue(np.array_equal(pc,boundary_codes(P)))
   po=list(map(tuple,np.argwhere(p)));to=list(map(tuple,np.argwhere(t)));sp=list(map(Fraction,spacing));tau2=Fraction(tau)**2
   def close(x,ys):return any(sum((sp[j]*(x[j]-y[j]))**2 for j in range(3))<=tau2 for y in ys)
   num=sum((w[int(pc[x])]for x in po if close(x,to)),Fraction(0))+sum((w[int(tc[x])]for x in to if close(x,po)),Fraction(0))
   return num/den if den else Fraction(1)
  for mode in range(3):
   A=np.zeros((5,5,5),bool);A[1:3,1:3,1:3]=True;B=np.roll(A,1,axis=0)
   g=[[A,np.roll(A,1,axis=1)],[B,np.roll(B,1,axis=2),A|B]]
   spacing=(.8,1.2,1.7);tau=[0.,.8,1.3][mode]
   q=compile_pair(A,B,g,spacing,tau);a,b=q['a'],q['b'];r=search(a,b,seconds=30);self.assertTrue(r['complete']);verify(a,b,r)
   vals=[]
   for z in itertools.product(range(2),range(3)):
    T=g[0][z[0]]|g[1][z[1]];v=direct(A,T,spacing,tau)-direct(B,T,spacing,tau)
    self.assertEqual(v,score(a,b,z));vals.append(v)
   self.assertEqual(min(vals),Fraction(r['min']));self.assertEqual(max(vals),Fraction(r['max']))
if __name__=='__main__':unittest.main()
