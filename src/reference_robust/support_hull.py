"""Exact upper envelope of integer lines. Standard convex-hull preprocessing.

No floating-point decisions. A line is (slope, intercept); queries use x=n/d,
d>0, and return d*intercept+n*slope without division.
"""
def upper_hull(lines):
    best={}
    for slope,intercept in lines:
        best[slope]=max(intercept,best.get(slope,intercept))
    hull=[]
    for line in sorted(best.items()):
        while len(hull)>=2:
            (a,b),(c,d)=hull[-2:];e,f=line
            # crossing(a,c) >= crossing(c,e): middle line never uniquely wins
            if (b-d)*(e-c)>=(d-f)*(c-a):hull.pop()
            else:break
        hull.append(line)
    return hull

def query(hull,n,d):
    if not hull or d<=0:raise ValueError('Empty hull or non-positive ratio denominator')
    lo,hi=0,len(hull)-1
    while lo<hi:
        mid=(lo+hi)//2
        a,b=hull[mid];c,e=hull[mid+1]
        if n*(c-a)+d*(e-b)>=0:lo=mid+1
        else:hi=mid
    a,b=hull[lo]
    return n*a+d*b
