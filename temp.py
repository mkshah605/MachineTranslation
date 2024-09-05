import torch as t

if t.backends.mps.is_available():
    mps_device = t.device("mps")
    x = t.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")