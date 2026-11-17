import argparse,csv,json
from fractions import Fraction
from collections import defaultdict

def evaluate(rows,margin):
    groups=defaultdict(dict)
    for r in rows:
        key=(r["case"],r["reference"])
        if key[1] in groups[key[0]]:raise ValueError("Duplicate case/reference")
        a,b=Fraction(r["score_a"]),Fraction(r["score_b"])
        if not(0<=a<=1 and 0<=b<=1):raise ValueError("Scores outside [0,1]")
        groups[key[0]][key[1]]=a-b
    if not groups:raise ValueError("No supplied references")
    out=[]
    for case,rr in sorted(groups.items()):
        lo,zlo=min((v,k)for k,v in rr.items());hi,zhi=max((v,k)for k,v in rr.items())
        c="A" if lo>margin else "B" if hi<-margin else "REVERSAL" if lo<-margin and hi>margin else "UNRESOLVED"
        out.append(dict(case=case,references=len(rr),minimum=str(lo),maximum=str(hi),minimum_reference=zlo,maximum_reference=zhi,decision=c,scope="supplied common-reference records only"))
    return out

def main():
    p=argparse.ArgumentParser(description="Exact paired decisions from common-reference scores in [0,1]")
    p.add_argument("csv");p.add_argument("--margin",default="1/100");a=p.parse_args();margin=Fraction(a.margin)
    if margin<0:p.error("margin must be nonnegative")
    with open(a.csv,newline="") as f:out=evaluate(csv.DictReader(f),margin)
    print(json.dumps(out,indent=2))
