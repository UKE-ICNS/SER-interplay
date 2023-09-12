#Here are the numerical examples mentioned in the paper
import numpy as np
import datetime
import torch as th
import pickle
import itertools
from funcs import attrs
from funcs import fam
from funcs import famd1
from funcs import famd2
from funcs import loadall
from funcs import un_roll1
from SER_interplay import SERmodel_multneuro_buf_c

#%% Network of 4 nodes
print(f'Network of 4 nodes') 
#Create all positive connections between devices and triangles
matr = 5*np.ones((4,4))

matr_4 = matr.astype(np.int8)
#device
matr_4[0,1] = 1
matr_4[1,2] = 1
matr_4[2,0] = 1
matr_4[0,3] = 1
matr_4[3,2] = -1
#diagonal - set to 0, no self-loops!
matr_4[0,0] = 0
matr_4[1,1] = 0
matr_4[2,2] = 0
matr_4[3,3] = 0

#define particular 4 node matrices
matr_f1 = matr_4.astype(np.int8)
matr_f2 = matr_4.astype(np.int8)

matr_f1[1,3] = 1
matr_f2[2,3] = 1

matr_f11 = matr_f1.astype(np.int8)
matr_f21 = matr_f2.astype(np.int8)

matr_f11[3,0] = -1
matr_f21[3,1] = -1

matr_f111 = matr_f11.astype(np.int8)
matr_f211 = matr_f21.astype(np.int8)
matr_f111[2,3] = 1
matr_f211[1,3] = 1

matr_f1111 = matr_f111.astype(np.int8)
matr_f2111 = matr_f211.astype(np.int8)
matr_f1111[3,1] = -1
matr_f2111[3,0] = -1

#adjacency matriix families
matrs = fam(matr_4,7)
matrf1 = fam(matr_f1,6)
matrf2 = fam(matr_f2,6)

matrf11 = fam(matr_f11,5)
matrf21 = fam(matr_f21,5)

matrf111 = fam(matr_f111,4)
matrf211 = fam(matr_f211,4)

matrf1111 = fam(matr_f1111,3)
matrf2111 = fam(matr_f2111,3)
#%% run all on GPU
ia = list(itertools.product([0,1,-1], repeat=4))
Cs = np.array(matrs.transpose((0,2,1)))
ress_buf = SERmodel_multneuro_buf_c(Cs, 100, ia)
name2 = f'Files\\data\\all_4devices_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file = open(name2, 'wb')

th.save(ress_buf,file)

loaded_buf2 = th.load(name2)

loaded_sims = loaded_buf2.cpu().detach().numpy()
#%% get attractors

#save attractor ends
name_at = f'Files//data//attractors4_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at = open(name_at, 'wb')

for k in range(len(loaded_sims)):
    
    res_attrs = attrs(loaded_sims[k],4)

    pickle.dump([res_attrs], file_at)

file_at.close()

# get unique attractors
# to load the existing dataset
items_attr = loadall(name_at)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_space, name_counts = un_roll1(attrs1,4)

items_attr = loadall(name_counts)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts = np.squeeze(c_ar_at)

items_attr1 = loadall(name_space)   
c_at1 = list(items_attr1)
c_ar_at1=np.array(c_at1)
space = np.squeeze(c_ar_at1)

#%% calculate when the device is not working
ph1 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph2 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 1., -1.,  0.]]) 

ph3 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.]]) 

ph4 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  0.,  0.]]) 


counter_never=0
counter_two=0
counter_one=0
k = 0
for i in range(len(matrs)):
    inter_count=0
    for el in space[i]:
        if (el == ph1).all():
            inter_count+=1
        if (el == ph2).all():
            inter_count+=1
        if (el == ph3).all():
            inter_count+=1
        if (el == ph4).all():
            inter_count+=1
    if inter_count==0:
        counter_never+=1
    else:
        k = np.append(k,i)
  
print(f'Cases when device is working: {counter_never}') 
print(f'Cases when device is not working: {3**7 - counter_never}') 

#%% Prediction
#The number of non-working devices:
nnw= np.concatenate((matrf2,matrf1))
nnw_big = np.unique(nnw,axis=0)
nnw_sm = np.unique(np.concatenate((matrf211,matrf111)),axis=0)

#create a set
nnw_big_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    nnw_big])
nnw_sm_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    nnw_sm])
f11_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    matrf11])
f21_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    matrf21])
f1111_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    matrf1111])
f2111_s = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    matrf2111])

fin = (nnw_big_s - f11_s.union(f21_s)).union(nnw_sm_s) - f1111_s.union(f2111_s)
print(f'Prediction: Cases when device is working: {3**7 - len(fin)}')
print(f'Prediction: Cases when device is not working: {len(fin)}')

#%% Common edge case
print(f'Network of 4 nodes, common edge') 
#Create all positive connections between 2 triangles
matr4 = np.array([[0,1,0,5],[0,0,1,5],[1,0,0,1],[0,1,0,0]])
matrs4 = fam(matr4,2)

#%% run all on GPU
ia4 = list(itertools.product([0,1,-1], repeat=4))
Cs4 = np.array(matrs4.transpose((0,2,1)))
ress_buf4 = SERmodel_multneuro_buf_c(Cs4, 100, ia4)
name4 = f'Files\\data\\all_4devices_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file4 = open(name4, 'wb')

th.save(ress_buf4,file4)

loaded_buf24 = th.load(name4)

loaded_sim4 = loaded_buf24.cpu().detach().numpy()

#%% get attractors
#save attractor ends
name_at4 = f'Files\\data\\attractors4_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at4 = open(name_at4, 'wb')

for k in range(len(loaded_sim4)):
    
    res_attrs4 = attrs(loaded_sim4[k],4)

    pickle.dump([res_attrs4], file_at4)

file_at4.close()

#get unique attractors
# to load the existing dataset
items_attr = loadall(name_at4)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_sp, name_c = un_roll1(attrs1,4)

items_attr = loadall(name_c)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts4 = np.squeeze(c_ar_at)

items_attr = loadall(name_sp)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
space4 = np.squeeze(c_ar_at)

#%% calculate when the triangles always present
ph4 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ -1., 0.,  1.]])

counter_never4=0
counter_two4=0

for i in range(len(matrs4)):
    inter_count=0
    for el in space4[i]:
        if (el == ph4).all():
            inter_count+=1
    if inter_count==1:
        counter_never4+=1
    if inter_count==0:
        counter_two4+=1

print(f'Cases when 1 phase is still there: {counter_never4}') 
print(f'Cases all triangles are damaged: {3**2 - counter_never4}') 

#%% Network of 5 nodes
print(f'Network of 5 nodes') 
#Create all positive connections between 2 triangles
matr_tr1 = np.array([[0,1,0],[0,0,1],[1,0,0]]) #random working triangle 1
matr_tr2 = matr_tr1.astype(np.int8) #random working triangle 2
matr0 = np.zeros((3,3))
matr1 = np.ones((3,3))

matr_up = np.hstack((matr_tr1[:,:-1], matr1))
matr_down = np.hstack((matr0[:,:-1], matr_tr2))

matr = np.vstack((matr_up,matr_down[1:]))

matr_5 = matr.astype(np.int8)
matr_5[0,3:] = 5
matr_5[1,3:] = 5
matr_5[2,4] = 5

matrs = fam(matr_5,5)

#%% run all on GPU
ia = list(itertools.product([0,1,-1], repeat=5))
Cs = np.array(matrs.transpose((0,2,1)))
ress_buf = SERmodel_multneuro_buf_c(Cs, 100, ia)
name2 = f'Files\\data\\all_5devices_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file = open(name2, 'wb')

th.save(ress_buf,file)

loaded_buf2 = th.load(name2)

loaded_sims = loaded_buf2.cpu().detach().numpy()

#%% get attractors
#save attractor ends
name_at = f'Files//data//attractors5_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at = open(name_at, 'wb')

for k in range(len(loaded_sims)):
    
    res_attrs = attrs(loaded_sims[k],5)

    pickle.dump([res_attrs], file_at)

file_at.close()

# get unique attractors
# to load the existing dataset
items_attr = loadall(name_at)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_sp, name_c = un_roll1(attrs1,5)

items_attr = loadall(name_c)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts = np.squeeze(c_ar_at)

items_attr = loadall(name_sp)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
space = np.squeeze(c_ar_at)

#%% calculate when the triangles always present
ph1 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

counter_never=0
counter_two=0
counter_one=0
for i in range(len(matrs)):
    inter_count=0
    for el in space[i]:
        if (el == ph1).all():
            inter_count+=1
    if inter_count==1:
        counter_never+=1
    if inter_count==0:
        counter_two+=1
 
print(f'Cases when triangles are working: {counter_never}') 
print(f'Cases when all triangles are damaged: {3**5 - counter_never}') 

#%% Cases when all the triangles will be gone analytically
# This triangles can work only if nodes 0 and 3 are firing simultaneously.
# There shouldn't be any negative connection from 0 to 4.
# The cases like that are all the possible matrices where there is -1 in [0,4].
#It's 3**4 cases.

print(f'Prediction: Cases when all triangles are damaged: {3**4}')

#%% Network of 6 nodes
print(f'Network of 6 nodes') 
#Create all positive connections between 2 triangles
matr_tr1 = np.array([[0,1,0],[0,0,1],[1,0,0]]) #random working triangle 1
matr_tr2 = matr_tr1.astype(np.int8) #random working triangle 2
matr0 = np.zeros((3,3))
matr1 = np.ones((3,3))

matr_up = np.hstack((matr_tr1, matr1))
matr_down = np.hstack((matr0, matr_tr2))

matr = np.vstack((matr_up,matr_down))

#Delete 1 by 1 connection
dels = np.array(list(itertools.product([0,1,-1], repeat=9))) #there are 3 points 
    #where triangles could connect
matrs_dels = dels.reshape((3**9,3,3))

matrs = np.zeros((3**9,6,6))
for i in range(len(matrs)):
    matrs[i][0:3,0:3] = matr_tr1
    matrs[i][3:6,3:6] = matr_tr2
    matrs[i][3:6,0:3] = matr0
    matrs[i][0:3,3:6] = matrs_dels[i]

#%% run all on GPU
ia = list(itertools.product([0,1,-1], repeat=6))
Cs = np.array(matrs.transpose((0,2,1)))
ress_buf = SERmodel_multneuro_buf_c(Cs, 100, ia)
name2 = f'Files\\data\\all_6devices_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file = open(name2, 'wb')

th.save(ress_buf,file)

loaded_buf2 = th.load(name2)

loaded_sims = loaded_buf2.cpu().detach().numpy()

#%% get attractors
#save attractor ends
name_at = f'Files\\data\\attractors_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at = open(name_at, 'wb')

for k in range(len(loaded_sims)):
    
    res_attrs = attrs(loaded_sims[k],6)

    pickle.dump([res_attrs], file_at)

file_at.close()

#get unique attractors
# to load the existing dataset
items_attr = loadall(name_at)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_sp, name_c = un_roll1(attrs1,6)

items_attr = loadall(name_c)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts = np.squeeze(c_ar_at)

items_attr = loadall(name_sp)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
space = np.squeeze(c_ar_at)

#%% calculate when the triangles always present
ph1 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph2 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

ph3 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.]])

counter_never=0
counter_two=0
counter_one=0
for i in range(len(matrs)):
    inter_count=0
    for el in space[i]:
        if (el == ph1).all():
            inter_count+=1
        if (el == ph2).all():
            inter_count+=1
        if (el == ph3).all():
            inter_count+=1
    if inter_count==3:
        counter_never+=1
    if inter_count==2:
        counter_two+=1
    if inter_count==1:
        counter_one+=1
 
print(f'Cases when all triangles are damaged, in percent: {(3**9 - counter_never - counter_two - counter_one)/(3**9):.0%}') 

#%% Prediction network of 6 nodes
#Create all the posssible connections between 2 triangles
matrs = np.array(list(itertools.product([0,1,-1], repeat=9))) #there are 3 points 
    #where triangles could connect
matrs = matrs.reshape((3**9,3,3))

#connecting rows
res = [0]*len(matrs)

for i in range(len(matrs)):
    #sum up the -1 in rows and columns
    a = np.sum(matrs[i][0]==-1)
    b = np.sum(matrs[i][1]==-1)
    c = np.sum(matrs[i][2]==-1)
    rows = np.max([a,b,c])

    a1 = np.sum(matrs[i][:,0]==-1)
    b1 = np.sum(matrs[i][:,1]==-1)
    c1 = np.sum(matrs[i][:,2]==-1)
    cols = np.max([a1,b1,c1])

    res[i] = np.max([rows,cols])

zers = np.sum(np.array(res)==0)
ons = np.sum(np.array(res)==1)
tws = np.sum(np.array(res)==2)
ths = np.sum(np.array(res)==3)

#%% calculate really all the triangle cases with assumptions
adj1 = 5*np.ones((3,3))
#adjacency matriix families
un, all_un, tw_calc, on_calc = famd1(adj1)
un_1, all_un2 = famd2(adj1)

un1 = np.unique(np.concatenate((matrs[np.array(res)==3],un)),axis=0)

#how to calculate 1 and 2
set_tw_calc = set([tuple([tuple([i for i in j]) for j in k]) for k in tw_calc])
set_tw_cr = set([tuple([tuple([i for i in j]) for j in k]) for k in 
    matrs[np.array(res)==2]])
tw_notin3 = set_tw_cr - set_tw_calc

un_1_with3 = set([tuple([tuple([i for i in j]) for j in k]) for k in un_1])
with3 = set([tuple([tuple([i for i in j]) for j in k]) for k in un1])
un1_wo3 = un_1_with3 - with3
un2 = tw_notin3.union(un1_wo3)

print(f'Prediction: When triangles are destroyed: {len(un1)/(3**9):.0%}')

#%% Network of 7 nodes
print(f'Network of 7 nodes') 

#Create all positive connections between 2 triangles
matr_tr1 = np.array([[0,1,0],[0,0,1],[1,0,0]]) #random working triangle 1
matr_tr2 = matr_tr1.astype(np.int8) #random working triangle 2
matr0 = np.zeros((3,3))
matr1 = np.zeros((3,3))

matr_up = np.hstack((matr_tr1, matr1))
matr_down = np.hstack((matr0, matr_tr2))

matr = np.vstack((matr_up,matr_down))

matr = np.insert(matr, 3, np.array((0,0,0,0,0,0)), 0) 
matr = np.insert(matr, 3, np.array((0,0,0,0,0,0,0)), 1) 

matr[0,3] = 1
matr[3,2] = -1

matr_5 = matr.astype(np.int8)
matr_5[4:,3] = 5

matrs = fam(matr_5,3)

#%% run all on GPU
ia = list(itertools.product([0,1,-1], repeat=7))
Cs = np.array(matrs.transpose((0,2,1)))
ress_buf = SERmodel_multneuro_buf_c(Cs, 100, ia)
name2 = f'Files\\data\\all_7devices_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file = open(name2, 'wb')

th.save(ress_buf,file)

loaded_buf2 = th.load(name2)

loaded_sims = loaded_buf2.cpu().detach().numpy()

#%% get attractors
#save attractor ends
name_at = f'Files//data//attractors7_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at = open(name_at, 'wb')

for k in range(len(loaded_sims)):
    
    res_attrs = attrs(loaded_sims[k],7)

    pickle.dump([res_attrs], file_at)

file_at.close()

# get unique attractors
# to load the existing dataset
items_attr = loadall(name_at)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_sp, name_c = un_roll1(attrs1,7)

items_attr = loadall(name_c)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts = np.squeeze(c_ar_at)

items_attr = loadall(name_sp)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
space = np.squeeze(c_ar_at)

#%% calculate when the triangles always present
ph1 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  0.,  0.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph11 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph12 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 1.,  -1., 0.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph13 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.]])

ph2 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  0.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

ph21 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

ph22 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 1.,  -1., 0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

ph23 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.]])

ph3 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  0.,  0.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.]])

ph31 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 0.,  1., -1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.]])

ph32 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ 1., -1.,  0.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.]])

ph33 = np.array([[-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.],
                [ -1., 0.,  1.],
                [-1.,  0.,  1.],
                [ 1., -1.,  0.],
                [ 0.,  1., -1.]])

counter_never=0
counter_two=0
counter_one=0
for i in range(len(matrs)):
    inter_count=0
    inter_count1=0
    inter_count2=0
    inter_count3=0
    for el in space[i]:
        if ((el == ph1).all() or (el == ph11).all() or (el == ph12).all() or (el == ph13).all()):
            inter_count1=1
        if ((el == ph2).all() or (el == ph21).all() or (el == ph22).all() or (el == ph23).all()):
            inter_count2=1
        if ((el == ph3).all() or (el == ph31).all() or (el == ph32).all() or (el == ph33).all()):
            inter_count3=1
    inter_count = inter_count1 + inter_count2 + inter_count3
    if inter_count==3:
        counter_never+=1
    if inter_count==2:
        counter_two+=1
    if inter_count==1:
        counter_one+=1
        
print(f'Cases when all triangles are damaged: {3**3 - counter_never - counter_two - counter_one}') 