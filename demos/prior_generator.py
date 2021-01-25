import numpy as np
import sys
import math

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

qfile = sys.argv[1]
gtfile = sys.argv[2]
ndb = int(sys.argv[3])
ngen = int(sys.argv[4])

queries = fvecs_read(qfile)

dist = np.random.zipf(1.25, size = ngen)
print(dist[:10])
samples = dist[dist < np.shape(queries)[0]].astype(int)

oqueries = queries[samples]
fvecs_write(qfile + "-exp", oqueries)

gt = ivecs_read(gtfile)
ogt = gt[samples]
ivecs_write(gtfile + "-exp", ogt)

priors = np.zeros((ndb, 1), dtype=np.float)
priors_exp = np.zeros((ndb, 1), dtype=np.float)

total_gt = np.shape(ogt)[0]*np.shape(ogt)[1]

unique, cnts = np.unique(ogt.flatten(), return_counts=True)
for el, cnt in zip(unique, cnts):
    priors[el] = cnt / total_gt

for i in range(np.shape(ogt)[1]):
    unique, cnts = np.unique(ogt[:, i].flatten(), return_counts = True)
    for el, cnt in zip(unique, cnts):
        priors_exp[el] += (cnt / np.shape(ogt)[0]) * math.exp(-i)

print(priors[:10, 0])
print(max(priors))
print(max(priors_exp))
fvecs_write("priors-flat.fvecs", priors)
fvecs_write("priors-expfalloff.fvecs", priors_exp)