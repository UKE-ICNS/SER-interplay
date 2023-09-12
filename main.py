import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import itertools
import pandas as pd
import networkx as nx
import torch as th
from funcs import attrs
from funcs import loadall
from funcs import overlap
from funcs import positive_triag
from funcs import small_device
from funcs import lim_cyclesn_before
from funcs import un_roll1
from SER_interplay import SERmodel_multneuro_buf_c

#%% Triangle calculation 
#to use random connectome with the size 9 (other size involves changing the function)!
#gr_size = 9
#A_ran = np.random.randint(-1,2,size=(gr_size, gr_size)) #random adjacency

#to use cabessa connectome (application example)
A_ran = np.array([[0,1,0,0,0,0,0,0,0], [0,0,1,0,1,1,1,1,1], [0,-1,0,0,0,0,0,0,0], 
              [-1,-1,-1,0,0,0,0,0,0], [0,0,0,1,0,1,0,0,1], [0,0,-1,-1,-1,0,-1,-1,0], 
              [0,0,0,0,0,-1,0,0,0], [0,0,0,-1,0,-1,0,0,0], [1,1,1,0,1,0,1,1,0]]) #directed BG network 
              #(SC, Th, RTN, GPi/SNr, STN, GPe, D2, D1, Ctx) 

#to use cabessa connectome (PD state) (application example)
# A_ran = np.array([[0,1,0,0,0,0,0,0,0], [0,0,1,0,1,1,1,1,1], [0,-1,0,0,0,0,0,0,0], 
#               [-1,-1,-1,0,0,0,0,0,0], [0,0,0,1,0,1,0,0,1], [0,0,-1,-1,-1,0,-1,-1,0], 
#               [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [1,1,1,0,1,0,1,1,0]]) #directed BG network 
#               #(SC, Th, RTN, GPi/SNr, STN, GPe, D2, D1, Ctx) 

#to use cabessa connectome (STN-DBS state) (application example)
# A_ran = np.array([[0,1,0,0,0,0,0,0,0], [0,0,1,0,1,1,1,1,1], [0,-1,0,0,0,0,0,0,0], 
#               [-1,-1,-1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,-1,-1,-1,0,-1,-1,0], 
#               [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [1,1,1,0,1,0,1,1,0]]) #directed BG network 
#               #(SC, Th, RTN, GPi/SNr, STN, GPe, D2, D1, Ctx) 

#find cycles of length 3
G_ran = nx.from_numpy_matrix(A_ran,create_using=nx.DiGraph)
used_graph = G_ran
a = nx.simple_cycles(used_graph)
aa = list(a)
aa_3 = np.array([i for i in aa if len(i)==3], dtype=np.int8)

#load pickles with template connections
items = loadall('Files\\notworkdev.pckl')   
fin = tuple(items)[0][0]

items1 = loadall('Files\\phase2trtr.pckl')   
un2 = tuple(items1)[0]

items2 = loadall('Files\\phase3trtr.pckl')   
un1 = tuple(items2)[0]

##1. Triangles
if aa_3.size>0: #if there are any triangles
    plus_trig, pl_clc = positive_triag(aa_3, A_ran) #find positive triangles
    #ee, dd = lim_cyclesn_before(A_ran,pl_clc,len(A_ran)-3)
    devices = []
    pl_link = []
    if plus_trig>0: #if there are positive triangles
        ##2. Hooks
        devices, pl_link, hlp1 = small_device(A_ran,pl_clc)

        #determine hooks in a correct tuple order for dataframe to find if hooks are there
        if hlp1[0]==0:
            hoo = tuple('x')
        else:
            hoo = tuple([pl_link[:int(hlp1[0])]])
        c = 0
        for i in range(1,len(hlp1)):
            c+=int(hlp1[i-1])
            cn = c + int(hlp1[i])
            if cn==c:
                hoo +=tuple('x')
            else:
                hoo += tuple([pl_link[c:cn]])

        #determine if one of the hooks is working and if triangle works
        work = tuple('s')
        death = tuple('s')
        for j in range(len(pl_clc)):
            ar2 = pl_clc[j].astype(np.int64)
            strs = ["yes" for x in range(len(hoo[j]))]
            if hoo[j]!='x':
                for i in range(len(hoo[j])):
                    ar1 = hoo[j][i]
                    if ar1[1] not in ar2:
                        ar_fr = np.concatenate((np.roll(ar2,-int(
                            np.where((ar2==ar1[0]))[0])),[ar1[1]]))
                        if len(ar_fr)==4:
                            tr1 = A_ran[ar_fr][:,ar_fr].astype(np.int64)
                            tr2 = tr1.astype(np.int64)
                            tr2[0,0] = 0
                            tr2[1,1] = 0
                            tr2[2,2] = 0
                            tr2[3,3] = 0
                        if tuple(map(tuple, tr2)) in fin:
                            strs[i]='no'
                        else:
                            trs = list(pl_clc.astype(np.int64))
                            #del trs[j]
                            for h_c in range(len(trs)):
                                #indirect connections
                                l1 = list(nx.all_simple_paths(G_ran, source=trs[h_c][0], target=ar1[1]))
                                l2 = list(nx.all_simple_paths(G_ran, source=trs[h_c][1], target=ar1[1]))
                                l3 = list(nx.all_simple_paths(G_ran, source=trs[h_c][2], target=ar1[1]))
                                l_big = l1+l2+l3
                                sun = set(np.unique(pl_clc))
                                sun1 = set(trs[h_c])
                                tr_nodes = list(sun-sun1)
                                kk = []
                                for ii in l_big:
                                    for jj in range(len(tr_nodes)):
                                        if tr_nodes[jj] in ii:
                                            break
                                        else:
                                            if jj==len(tr_nodes)-1:  
                                                kk.append(ii)
                                if len(kk)!=0:
                                   strs[i]='ind_infl'
                                   break 
                                #direct connections
                                lin = A_ran[trs[h_c]][:,ar1[1]]
                                if not (lin==np.zeros(3)).all():
                                   strs[i]='infl'
                                   break 
                    else:
                        strs[i]='self-hook'
                death += tuple([strs])
                if 'yes' in strs:
                    work += tuple(['no'])
                else:
                    work += tuple(['yes'])
            else:
                death += tuple('x')
                work += tuple(['yes'])
        death = death[1:]
        work = work[1:]

        #print a dataframe for hooks
        data = {'Positive triangles': tuple(pl_clc), 'How many hooks': hlp1, 
        'Positive hook links': hoo, 'Working hooks': death, 'Triangle works': work}  
        tr_and_hooks = pd.DataFrame(data)  
        print('Triangles and their hooks')
        print(tr_and_hooks)

        #leave only working trinagles
        w_pl_clc = pl_clc[np.where((np.array(work)=='yes'))[0]]
        #to check if there are any triangles that are non-working due to other interactions
        pos_atr_fromw, rea_atr_fromw = lim_cyclesn_before(A_ran,w_pl_clc,len(A_ran)-3)

        ##3. Triangle-hook interactions

        ##4. Triangle overlap
        overlap_tr = overlap(w_pl_clc.astype(np.int8))
        #check stop
        if overlap_tr == []:
            print("No working triangles")
        else:
            print('')
            print('Triangle overlap') 
            tr_overlap = pd.DataFrame(overlap_tr.tolist(), columns =tuple(w_pl_clc))  
            tr_overlap.index = tuple(w_pl_clc)
            print(tr_overlap)
            
            #Define constraints
            strs1 = [["?" for x in range(len(w_pl_clc))] for z in range(len(w_pl_clc))]
            strs3 = [["?" for x in range(len(w_pl_clc))] for z in range(len(w_pl_clc))]
            noov1 = np.zeros(3)
            noov2 = np.zeros(3)
            np.fill_diagonal(overlap_tr, overlap_tr.diagonal() + 1)
            for p1 in range(len(w_pl_clc)):
                for p2 in range(len(w_pl_clc)):
                    if overlap_tr[p1][p2]==4:
                        strs1[p1][p2]='itself'
                        strs3[p1][p2]='itself'
                    if overlap_tr[p1][p2]==0 or overlap_tr[p1][p2]==3:
                        strs1[p1][p2]='no overlap'
                        strs3[p1][p2]='h'
                        noov1 = np.vstack((noov1,w_pl_clc[p1]))
                        noov2 = np.vstack((noov2,w_pl_clc[p2]))
                    if overlap_tr[p1][p2]==2:
                        #bypass direction
                        ch_order1 = np.append(w_pl_clc[p1],w_pl_clc[p1][0])
                        ch_order2 = np.append(w_pl_clc[p2],w_pl_clc[p2][0])
                        com = list(set(ch_order1).intersection(set(ch_order2)))
                        f1 = np.where((ch_order1==com[0]))[0][-1]
                        l1 = np.where((ch_order1==com[1]))[0][-1]
                        f2 = np.where((ch_order2==com[0]))[0][-1]
                        l2 = np.where((ch_order2==com[1]))[0][-1]
                        if l1>f1:
                            if l2>f2:
                                strs1[p1][p2]='phase constrained'
                                strs3[p1][p2]='phase constrained'
                            else:
                                strs1[p1][p2]='deactivation'
                                strs3[p1][p2]='deactivation'
                                # noov1 = np.vstack((noov1,w_pl_clc[p1]))
                                # noov2 = np.vstack((noov2,w_pl_clc[p2]))
                        if l1<f1:
                            if l2<f2:
                                strs1[p1][p2]='phase constrained'
                                strs3[p1][p2]='phase constrained'
                            else:
                                strs1[p1][p2]='deactivation'
                                strs3[p1][p2]='deactivation'
                                # noov1 = np.vstack((noov1,w_pl_clc[p1]))
                                # noov2 = np.vstack((noov2,w_pl_clc[p2]))
                    if overlap_tr[p1][p2]==1:
                        inter = np.intersect1d(w_pl_clc[p1],w_pl_clc[p2])
                        h1 = np.where(w_pl_clc[p1]==inter)[0]
                        h2 = np.where(w_pl_clc[p2]==inter)[0]
                        r1 = np.roll(w_pl_clc[p1], -h1-1)[0]
                        r2 = np.roll(w_pl_clc[p2], -h2)[-1]
                        checker = A_ran[r1][r2]
                        if checker!=-1:
                            strs1[p1][p2]='phase constrained'
                            strs3[p1][p2]='phase constrained'
                        else:
                            strs1[p1][p2]='deactivation'
                            strs3[p1][p2]='deactivation'

            print('')
            print('Triangle overlap constraints') 
            tr_overlap_constr = pd.DataFrame(strs1, columns =tuple(w_pl_clc))  
            tr_overlap_constr.index = tuple(w_pl_clc)
            print(tr_overlap_constr)

            ##5. Negative links pattern
            #print a dataframe for negative interactions for non-overlapping triangles
            if (noov1==np.zeros(3)).all():
                print('')
                print('All triangles overlap')
            else:
                noov1 = noov1[1:]
                noov2 = noov2[1:]
                strs2 = ["?" for x in range(len(noov2))]
                for l in range(len(noov2)):
                    frame = A_ran[noov1[l].astype(np.int32)][:,noov2[l].astype(np.int32)]
                    if not np.any(frame==-1):
                        #print(frame)
                        strs2[l] = 'intact triangles'
                    elif tuple(map(tuple, frame)) in un2:
                        strs2[l]='2 phases are destroyed'
                    elif tuple(map(tuple, frame)) in un1:
                        strs2[l]='deactivation'
                    else:
                        strs2[l] = '1 phase is destroyed'
                data1 = {'Triangle 1': tuple(noov1), 'Triangle 2': tuple(noov2), 
                'Interaction result for triangle 2': tuple(strs2)}  
                tr_tr = pd.DataFrame(data1)  
                print('')
                print('Triangle-Triangle interaction')
                print(tr_tr)

            print('')
            print('Triangles') 
            tr_fin = pd.DataFrame(strs3, columns =tuple(w_pl_clc))  
            tr_fin.index = tuple(w_pl_clc)
            if 'h' in tr_fin.values:
                tr_fin[tr_fin=='h']=strs2
            print(tr_fin)
            for i in range(len(w_pl_clc)):
                for j in range(len(w_pl_clc)):
                    if tr_fin.iloc[i,j]=='deactivation' or tr_fin.iloc[i,j]=='itself':
                        tr_fin.iloc[i,j] = 'dd'
            fin_w_pl_clc = w_pl_clc.astype(np.int64)
            for k in range(len(w_pl_clc)):
                if (tr_fin.iloc[:,k] == 'dd').all():
                    fin_w_pl_clc[k]=[0,0,0]
            fin_w_pl_clc = fin_w_pl_clc.tolist()
            while [0, 0 ,0] in fin_w_pl_clc:
                fin_w_pl_clc.remove([0, 0 ,0])
            fin_w_pl_clc = np.array(fin_w_pl_clc)
            if len(w_pl_clc)!=0:
                if len(fin_w_pl_clc)!=0:
                    pos_atr, rea_atr = lim_cyclesn_before(A_ran,fin_w_pl_clc,len(A_ran)-3)
                else:
                    print('All triangles are deactivated or itself')
                    pos_atr, rea_atr = lim_cyclesn_before(A_ran,w_pl_clc,len(A_ran)-3)
            else:
                pos_atr, rea_atr = lim_cyclesn_before(A_ran,w_pl_clc,len(A_ran)-3)
        
    else:
        print('There are no positive triangles in this graph')
else:
        print('There are no positive triangles in this graph')

#%% run all on GPU to check if we get the same attractors by simulation
As = np.array((A_ran,A_ran)) #constracting Cs arrays for a function
ia = list(itertools.product([0,1,-1], repeat=9))
Cs = np.array(As.transpose((0,2,1))) #constracting Cs arrays for a function
ress_buf = SERmodel_multneuro_buf_c(Cs, 100, ia)
name2 = f'Files\\data\\ser_data_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file = open(name2, 'wb')

th.save(ress_buf,file) #saving

loaded_buf2 = th.load(name2)

loaded_sims = loaded_buf2.cpu().detach().numpy() #loading to a numpy array
#%% get attractors
#save attractor ends
name_at = f'Files//data//attractors_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
file_at = open(name_at, 'wb')

for k in range(len(loaded_sims)):
    
    res_attrs = attrs(loaded_sims[k])

    pickle.dump([res_attrs], file_at)

file_at.close()

#get unique attractors for all data chunks
# to load the existing dataset
items_attr = loadall(name_at)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
attrs1 = np.squeeze(c_ar_at)

name_space, name_counts = un_roll1(attrs1)

items_attr = loadall(name_counts)   
c_at = list(items_attr)
c_ar_at=np.array(c_at)
counts = np.squeeze(c_ar_at)

items_attr1 = loadall(name_space)   
c_at1 = list(items_attr1)
c_ar_at1=np.array(c_at1)
space = np.squeeze(c_ar_at1)
