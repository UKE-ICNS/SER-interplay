import datetime
import itertools
import pickle
import numpy as np
from numba import jit
#%% all functions

#Function to calculate positive triangles
@jit(nopython=True, cache=True)
def positive_triag(aa_3, adj):
    '''Takes triangle sequencies and an adjacency matrix, 
    returns number of positive triangles and 
    an array of positive triangles'''
    pos=0 #number of positive cycles
    pos_clc=0

    cycles = aa_3
    whh = np.zeros(len(cycles), dtype=np.int8)

    for j in range(len(cycles)):
        cycle1 = np.zeros(len(cycles[j])+1, dtype=np.int8)
        cycle1[:3] = cycles[j]
        cycle1[-1] = cycles[j][0]
        sumw=0
        for i in range(1, len(cycle1)):
            sumw = sumw+adj[(cycle1[i-1], cycle1[i])]
        whh[j] = sumw

    wgth = whh
    pos = (wgth==3).sum()

    pos_clc = cycles[wgth==3] #pos cycles

    return pos, pos_clc

#Function to calculate small device
def small_device(A_ran, pos_clc):
    '''Takes an adjacency matrix and an array of positive triangles, 
    returns a device, a positive link in a device and 
    a node where the feedback loop is for a triangle'''
    A_us = A_ran #used adjacency matrix
    dev = np.zeros(2)
    hlp = np.zeros(1)
    pl_link = np.expand_dims(np.zeros(2),axis=0)
    for i in range(len(pos_clc)): 
        h=0
        for j in range(3):
            clcnow = np.roll(pos_clc, j, axis=1)[i]  #positive 3 node cycle in question

            #the 1 node in a cycle
            fr=clcnow[0].astype(np.int8)
            #the last node of the cycle
            ls = clcnow[2].astype(np.int8)

            #the nodes where the first node goes
            outs=np.where(A_us[fr]==1)[0]
            #the nodes which come to last node
            ins=np.where(A_us.T[ls]==-1)[0]

            interest = np.intersect1d(ins,outs)

            if interest.size>0:
                dev = np.append(dev,list(itertools.product(*[[pos_clc[i]],interest])))
                #print(f'The device is {dev} on a roll {j} for a positive cycle {i}')
            
            h += len(interest)

            for k in range(len(interest)):
                pl_link = np.append(pl_link,np.expand_dims(list([np.roll(pos_clc, j, axis=1)[i][0],interest[k]]),axis=0), axis=0)
        hlp = np.append(hlp,h)
    hlp = hlp[1:]
    dev = dev[2:]
    pl_link = pl_link[1:,:].astype(np.int8)
    return dev, pl_link, hlp

#Function to calculate triangle overlap and matrix of triangles with overlap
def overlap(pos_clc):
    '''Takes an array of positive triangles, returns an array of overlaps'''
    ov_matr = []
    if len(pos_clc)>0:
        ov_matr = np.zeros((len(pos_clc),len(pos_clc)))
        for i in range(len(pos_clc)):
            seti = set(pos_clc[i])
            for j in range(len(pos_clc)):
                setj = set(pos_clc[j])
                ov_matr[i,j] = len(seti.intersection(setj))
    return ov_matr

#Function to remove duplicates           
def Extract(lst):
    '''Extracts first appearances from list'''
    return [item[0] for item in lst]

#Function to find uniques
def Unique(lst):
    '''Finds unique items'''
    return [list(set(item)) for item in lst]

#Function to check if comething is in list
def in_list(c, classes):
    '''Returns -1 if c is contained in list'''
    for f, sublist in enumerate(classes):
        if c in sublist:
            return f
    return -1

#Function to get all unique attractors
def un_roll(atr_list, s):
    '''Takes ends of an unpickled array, returns attractors of the system'''
    u_a = np.unique(atr_list,axis=0)
    rollers = [[]]

    #need to check if in u_a any attractors which are just rolled versions of themselves
    for i in range(len(u_a)):
        for j in range(len(u_a)):
            if np.sum(u_a[i] == np.roll(u_a[j],1,axis=1))==3*s: #check rolled array
                k=in_list(i, rollers)
                q=in_list(j, rollers)
                if (k==-1) and (q==-1):
                    rollers.append([i,j])
                if (k==-1) and (q!=-1):
                    rollers[q].append(i)
                    rollers.append([])
                if (k!=-1) and (q==-1):
                    rollers[k].append(j)
                    rollers.append([])
                if (k!=-1) and (q!=-1):
                    # rollers[q].append(i)
                    # rollers[k].append(j)
                    rollers.append([])
        t=in_list(i, rollers)
        if t==-1:
            rollers.append([i])

    #remove empty lists
    rolled = [x for x in rollers if x != []]           

    fin = u_a[Extract(rolled),:,:]

    return fin

#Function to get all unique attractors for chunked data
def un_roll1(atr_list, s):
    '''Takes ends of an unpickled array for all chunks, returns pickle files 
    containing attractors of the system and their amount for all chunks.
    s=size of the network'''
    name_sp = f'Files//data//space_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
    file_sp = open(name_sp, 'wb')

    name_c = f'Files//data//counts_{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")}.pckl'  
    file_c = open(name_c, 'wb')

    for counter in range(len(atr_list)):  
        attrs = atr_list[counter]

        no_at = 0 #no attractor of size 3 is defined
        fix_p0 = 0 #fixed point 0
        other = 0 #attractor of size 3
        indices_help = []

        for i in range(np.shape(attrs)[0]):
            if ((attrs[i]==1).all())==1: no_at += 1
            if ((attrs[i]==0).all())==1: fix_p0 += 1
            if (((attrs[i]==1).all())!=1) and ((attrs[i]==0).all())!=1: 
                other +=1
                indices_help.append(i)

        #create an arrray of all the attractors of size 3
        all_attractors = attrs[indices_help]
        u_a = np.unique(all_attractors,axis=0)
        u_a_counts = np.unique(all_attractors,axis=0,return_counts=True)[-1]
        rollers = [[]]

        #need to check if in u_a any attractors which are just rolled versions of themselves
        for i in range(len(u_a)):
            for j in range(len(u_a)):
                if np.sum(u_a[i] == np.roll(u_a[j],1,axis=1))==3*s: #check rolled array
                    k=in_list(i, rollers)
                    q=in_list(j, rollers)
                    if (k==-1) and (q==-1):
                        rollers.append([i,j])
                    if (k==-1) and (q!=-1):
                        rollers[q].append(i)
                        rollers.append([])
                    if (k!=-1) and (q==-1):
                        rollers[k].append(j)
                        rollers.append([])
                    if (k!=-1) and (q!=-1):
                        # rollers[q].append(i)
                        # rollers[k].append(j)
                        rollers.append([])
            t=in_list(i, rollers)
            if t==-1:
                rollers.append([i])

        #remove empty lists
        rolled = [x for x in rollers if x != []]           

        fin = u_a[Extract(rolled),:,:]

        fin_counts = np.zeros(len(fin))

        for i in range(len(fin)):
            fin_counts[i] = np.sum(u_a_counts[Unique(rolled)[i]])

        pickle.dump([fin], file_sp)
        pickle.dump([fin_counts], file_c)

    file_c.close()
    file_sp.close()
    
    return[name_sp, name_c]

#Function to load pickles
def loadall(filename):
    '''Takes a filename, loads pickles'''
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#Function to get the limit cycles
def lim_cyclesn_before(A_ran, pos_clc, n): #n=matrix size - 3
    '''Takes an adjacency matrix, positive triangles and matrix size minus 3, 
    loads all attractors that could have been and the ones that are actually there'''
    pos_atr = np.zeros(((4**n)*len(pos_clc),len(A_ran),3))
    #create all possible limit cycles
    for i in range(len(pos_clc)):
        pos_atr[(4**n)*i:(4**n)*(i+1),pos_clc[i][0].astype(np.int8),:] = [1,-1,0]
        pos_atr[(4**n)*i:(4**n)*(i+1),pos_clc[i][1].astype(np.int8),:] = [0,1,-1]
        pos_atr[(4**n)*i:(4**n)*(i+1),pos_clc[i][2].astype(np.int8),:] = [-1,0,1]
        ninth = np.arange(0,(n+3))
        numb = [x for x in ninth if x not in [pos_clc[i][0].astype(np.int8),pos_clc[i][1].astype(np.int8),pos_clc[i][2].astype(np.int8)]]
        pos_atr[(4**n)*i:(4**n)*(i+1),numb,:] = np.array(list(itertools.product([[0,1,-1],[1,-1,0],[-1,0,1],[0,0,0]], repeat=n)))

    rea_atr = np.expand_dims(np.zeros_like(pos_atr[0]), axis=0)

    #delete all which can't be
    for i in range(len(pos_atr)):
        atr = pos_atr[i,:,:].astype(np.int8)
        ch1 = 0

        for j in range(3):
            #rolling four attrs
            attr1 = np.roll(atr,j,axis=1)
            
            #all points which fire in attr
            from1 = np.where(attr1[:,0]==1)[0]
            fire1 = np.intersect1d(np.where(np.sum(A_ran[from1,:],axis=0)>0)[0],
                np.where(attr1[:,0]==0)[0])

            #all points where fire in attr
            to1 = np.where(attr1[:,1]==1)[0]

            if not set(to1) == set(fire1):
                ch1+=1
    
        if ch1==0:
            rea_atr = np.append(rea_atr,np.expand_dims(atr,axis=0),axis=0)

    rea_atr = un_roll(rea_atr, len(A_ran))
    
    for i in range(len(rea_atr)):
        if np.all(rea_atr[i]==0):
            rea_atr = list(rea_atr)
            del rea_atr[i]  
            break

    return pos_atr, rea_atr

#Function to get the repeating ends from the initial conditions
@jit(nopython=True, cache=True)
def attrs(chunk, s):
    '''Gets the repeating ends of the array, s=size of the graph'''
    c_ar_sq_us = chunk #comment in and out for different conditions

    steps_with_effects = 0 #transient period
    at_s=3 #attractor size
    attrs = np.ones((np.shape(c_ar_sq_us)[0],s,3))

    for i in range(np.shape(c_ar_sq_us)[0]):
        loop_step = steps_with_effects
        while loop_step<=(np.shape(c_ar_sq_us)[2]-2*at_s):
            attractor = c_ar_sq_us[i][:,loop_step:loop_step+at_s]
            attractor_shift = c_ar_sq_us[i][:,loop_step+at_s:loop_step+2*at_s]
            loop_test = (attractor==attractor_shift)
            if loop_test.all()!=1: 
                print("No stable attractor for condition ", i)
                break
            if loop_step == (np.shape(c_ar_sq_us)[2]-2*at_s):
                at = attractor
                attrs[i] = at
            loop_step += 1

    return attrs

#Function to create families of adjacency matrices
def fam(adj, empt):
    '''Gets adjacency matrix template and the number of empty spaces, 
    returns a family of adjacency matrices'''
    q = np.expand_dims(adj,axis=0)
    q1 = np.repeat(q,3**empt,axis=0)
    iters = list(itertools.product([0,1,-1], repeat=empt))
    for i in range(len(iters)):
        q1[i][q1[i]==5] = iters[i]
    return q1

#Function to create more complicated families of adjacency matrices
def famd1(adjx):
    '''Gets adjacency matrix template, returns a family of adjacency matrices'''
    adjx1 = adjx.astype(np.int8)
    adjx2 = adjx.astype(np.int8)
    adjx3 = adjx.astype(np.int8)
    adjx4 = adjx.astype(np.int8)
    adjx5 = adjx.astype(np.int8)
    adjx6 = adjx.astype(np.int8)
    adjx7 = adjx.astype(np.int8)
    adjx8 = adjx.astype(np.int8)
    adjx9 = adjx.astype(np.int8)

    adjx21 = adjx.astype(np.int8)
    adjx22 = adjx.astype(np.int8)
    adjx23 = adjx.astype(np.int8)
    adjx24 = adjx.astype(np.int8)
    adjx25 = adjx.astype(np.int8)
    adjx26 = adjx.astype(np.int8)
    adjx27 = adjx.astype(np.int8)
    adjx28 = adjx.astype(np.int8)
    adjx29 = adjx.astype(np.int8)

    adjx10 = adjx.astype(np.int8)
    adjx11 = adjx.astype(np.int8)
    adjx12 = adjx.astype(np.int8)

    adjx1[0][0] = -1
    adjx1[1][0] = -1
    adjx1[1][2] = -1

    adjx2 = np.roll(adjx1,1,axis=0)
    adjx3 = np.roll(adjx1,2,axis=0)

    adjx4 = np.roll(adjx1,1,axis=1)
    adjx5 = np.roll(adjx1,2,axis=1)
    adjx6 = np.roll(adjx2,1,axis=1)
    adjx7 = np.roll(adjx2,2,axis=1)
    adjx8 = np.roll(adjx3,1,axis=1)
    adjx9 = np.roll(adjx3,2,axis=1)

    adjx21[0][0] = -1
    adjx21[1][0] = -1
    adjx21[0][1] = -1

    adjx22 = np.roll(adjx21,1,axis=0)
    adjx23 = np.roll(adjx21,2,axis=0)

    adjx24 = np.roll(adjx21,1,axis=1)
    adjx25 = np.roll(adjx21,2,axis=1)
    adjx26 = np.roll(adjx22,1,axis=1)
    adjx27 = np.roll(adjx22,2,axis=1)
    adjx28 = np.roll(adjx23,1,axis=1)
    adjx29 = np.roll(adjx23,2,axis=1)

    adjx10[0][2] = -1
    adjx10[1][1] = -1
    adjx10[2][0] = -1

    adjx11 = np.roll(adjx10,1,axis=0)
    adjx12 = np.roll(adjx10,2,axis=0)

    #create families of matrices and find uniques
    famx1 = fam(adjx1, 6)
    famx2 = fam(adjx2, 6)
    famx3 = fam(adjx3, 6)
    famx4 = fam(adjx4, 6)
    famx5 = fam(adjx5, 6)
    famx6 = fam(adjx6, 6)
    famx7 = fam(adjx7, 6)
    famx8 = fam(adjx8, 6)
    famx9 = fam(adjx9, 6)

    famx10 = fam(adjx10, 6)
    famx11 = fam(adjx11, 6)
    famx12 = fam(adjx12, 6)

    famx21 = fam(adjx21, 6)
    famx22 = fam(adjx22, 6)
    famx23 = fam(adjx23, 6)
    famx24 = fam(adjx24, 6)
    famx25 = fam(adjx25, 6)
    famx26 = fam(adjx26, 6)
    famx27 = fam(adjx27, 6)
    famx28 = fam(adjx28, 6)
    famx29 = fam(adjx29, 6)

    all_famx = np.concatenate((famx1,famx2,famx3,famx4,famx5,famx6,
        famx7,famx8,famx9,famx10,famx11,famx12,famx21,famx22,famx23,famx24,famx25,
        famx26,famx27,famx28,famx29))

    famx_un = np.unique(all_famx,axis=0)

    tw = np.unique(np.concatenate((famx1,famx2,famx3,famx4,famx5,famx6,
        famx7,famx8,famx9,famx21,famx22,famx23,famx24,famx25,
        famx26,famx27,famx28,famx29)),axis=0)

    on = np.unique(np.concatenate((famx10,famx11,famx12)),axis=0)

    return famx_un, all_famx, tw, on

#Function to create another more complicated families of adjacency matrices
def famd2(adjx):
    '''Gets adjacency matrix template, returns a family of adjacency matrices'''
    adjx1 = adjx.astype(np.int8)
    adjx2 = adjx.astype(np.int8)
    adjx3 = adjx.astype(np.int8)
    adjx4 = adjx.astype(np.int8)
    adjx5 = adjx.astype(np.int8)
    adjx6 = adjx.astype(np.int8)
    adjx7 = adjx.astype(np.int8)
    adjx8 = adjx.astype(np.int8)
    adjx9 = adjx.astype(np.int8)

    adjx1[0][0] = -1
    adjx1[1][2] = -1

    adjx2 = np.roll(adjx1,1,axis=0)
    adjx3 = np.roll(adjx1,2,axis=0)

    adjx4 = np.roll(adjx1,1,axis=1)
    adjx5 = np.roll(adjx1,2,axis=1)
    adjx6 = np.roll(adjx2,1,axis=1)
    adjx7 = np.roll(adjx2,2,axis=1)
    adjx8 = np.roll(adjx3,1,axis=1)
    adjx9 = np.roll(adjx3,2,axis=1)

    #create families of matrices and find uniques
    famx1 = fam(adjx1, 7)
    famx2 = fam(adjx2, 7)
    famx3 = fam(adjx3, 7)
    famx4 = fam(adjx4, 7)
    famx5 = fam(adjx5, 7)
    famx6 = fam(adjx6, 7)
    famx7 = fam(adjx7, 7)
    famx8 = fam(adjx8, 7)
    famx9 = fam(adjx9, 7)

    all_famx = np.concatenate((famx1,famx2,famx3,famx4,famx5,famx6,
        famx7,famx8,famx9))

    famx_un = np.unique(all_famx,axis=0)

    return famx_un, all_famx