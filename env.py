import numpy as np
import sys
import random
import math
import copy

class env_class(object):

    def step(self,a,L,Assemble_time,C,time_start,MPS):
        MPS_ = copy.deepcopy(MPS)
        MPS_[a] = MPS[a] - 1
        r1=0
        for i in range(len(L)):
            # b = max(C-(Assemble_time[i][a]+time_start[i]),0)
            c = max((Assemble_time[i][a]+time_start[i])-L[i],0)
            r1=r1+c
            if Assemble_time[i][a]+time_start[i]-L[i] >0:
                time_start[i] = L[i]-C
            else:
                time_start[i] = max(Assemble_time[i][a]+time_start[i]-C,0)
        state=np.append(np.array(MPS_),np.array(time_start))
        return state,r1,time_start,MPS_


