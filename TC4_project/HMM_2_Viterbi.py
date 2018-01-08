from pandas import Series, DataFrame
import pandas as pd
import pickle
import numpy as np

#address of dossier
adr_test10 = './data/test10.pkl'
adr_train10 = './data/train10.pkl'
adr_test20 = './data/test20.pkl'
adr_train20 = './data/train20.pkl'

#read in the data
test10 = pickle.load(open(adr_test10, "rb"))
train10 = pickle.load(open(adr_train10, "rb"))
test20 = pickle.load(open(adr_test20, "rb"))
train20 = pickle.load(open(adr_train20, "rb"))
voc_tag = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


#caculate the parameter of HMM: A,B,p(initial distribution of tag)
def calcule_Parametres(mydata):
   #list de tag
    state_list = []
    for i in range(len(mydata)):
        for j in range(len(mydata[i])):
            state_list.append(mydata[i][j][1])
    state_list = list(set(state_list))
    #vocabulaire
    vocabulaire = []
    for i in range(len(mydata)): 
        for j in range(len(mydata[i])):
            vocabulaire.append(mydata[i][j][0])
    
    vocabulaire = list(set(vocabulaire))
    
    #calculate parameters N1,N2,N3,C0,C1,C2,NW2,NW3
    N1C1 = Series(np.zeros(len(state_list)),index=state_list)
    N2C2 = DataFrame(np.zeros([len(state_list),len(state_list)]),columns=state_list, index=state_list)
    N3 = DataFrame(columns=state_list, index=range(1))
    for index in state_list:
        N3[index][0] = DataFrame(np.zeros([len(state_list),len(state_list)]),columns=state_list, index=state_list)
    C0 = 0
    NW = Series(np.zeros(len(vocabulaire)),index=state_list)
    
    
    
    NW3 = DataFrame(columns=vocabulaire, index=range(1))
    for voc in vocabulaire:
        NW3[voc][0] = DataFrame(np.zeros([len(state_list),len(state_list)]),columns=state_list, index=state_list)
    
    #calcule the distribution for pi
    distribution_de_tag_initial = {}
    for index in state_list:
        distribution_de_tag_initial[index] = 0
    for i in range(len(mydata)):
        distribution_de_tag_initial[mydata[i][0][1]] += 1
    for index in state_list:
        distribution_de_tag_initial[index] /= float(len(mydata))
    pi = Series(distribution_de_tag_initial)
     
    #calcule the matrix B_1
    B_1 = DataFrame(np.zeros([len(vocabulaire),len(state_list)]),columns=state_list, index=vocabulaire)
    for i in range(len(mydata)):
        for j in range(len(mydata[i])):
            B_1[mydata[i][j][1]][mydata[i][j][0]] += 1
    NW2 = B_1.copy(deep=True)
    for index in state_list:
        B_1[index] /= float(sum(B_1[index]))
    
    for i in range(len(mydata)):
        for j in range(len(mydata[i])):
            N1C1[mydata[i][j][1]] += 1
            NW[mydata[i][j][0]] += 1
            C0 += 1
            if len(mydata[i])-j == 1: continue
            
            N2C2[mydata[i][j][1]][mydata[i][j+1][1]] += 1
            NW3[mydata[i][j+1][0]][0][mydata[i][j][1]][mydata[i][j+1][1]] += 1    
                
            if len(mydata[i])-j-1 == 1: continue
            N3[mydata[i][j][1]][0][mydata[i][j+1][1]][mydata[i][j+2][1]] += 1
    
    #calculate a_ij
    A_1 = DataFrame(np.zeros([len(state_list),len(state_list)]),columns=state_list, index=state_list)
    for index_1 in state_list:
        for index_2 in state_list:
            k2 = (np.log(N2C2[index_1][index_2]+1)+1)/(np.log(N2C2[index_1][index_2]+1)+2)
            if N1C1[index_1]==0: temp1=0
            else: temp1 = k2*N2C2[index_1][index_2]/N1C1[index_1]
            temp2 = (1-k2)*N1C1[index_2]/C0
            A_1[index_1][index_2] = temp1 + temp2
    
    
    
    #calculate a_ijk
    A_ = DataFrame(columns=state_list,index=range(1))
    for index in state_list:
        A_[index][0] = DataFrame(np.zeros([len(state_list),len(state_list)]), columns=state_list, index=state_list)
    for index_1 in state_list:
        for index_2 in state_list:
            for index_3 in state_list:
                k2 = (np.log(N2C2[index_2][index_3]+1)+1)/(np.log(N2C2[index_2][index_3]+1)+2)
                k3 = (np.log(N3[index_1][0][index_2][index_3]+1)+1)/(np.log(N3[index_1][0][index_2][index_3]+1)+2)
                if N2C2[index_1][index_2] == 0: temp1=0
                else:  temp1 = k3*N3[index_1][0][index_2][index_3]/N2C2[index_1][index_2]
                if N1C1[index_2]==0: temp2=0
                else: temp2 = (1-k3)*k2*N2C2[index_2][index_3]/N1C1[index_2]
                temp3 = (1-k2)*(1-k3)*N1C1[index_3]/C0
                A_[index_3][0][index_1][index_2] = temp1 + temp2 + temp3
    
    #calculate bk_ij
    
    B_ = DataFrame(columns=vocabulaire, index=range(1))
    for voc in vocabulaire:
        B_[voc][0] = DataFrame(np.zeros([len(state_list),len(state_list)]), columns=state_list, index=state_list)
    for index_1 in state_list:
        for index_2 in state_list:
            for voc in vocabulaire:
                k3 = (np.log(NW3[voc][0][index_1][index_2]+1)+1)/(np.log(NW3[voc][0][index_1][index_2]+1)+2)
                if N2C2[index_1][index_2] == 0: temp1=0
                else: temp1 = k3*NW3[voc][0][index_1][index_2]/N2C2[index_1][index_2]
                if N1C1[index_2] == 0: temp2 = 0
                else: temp2 = (1-k3)*NW2[index_2][voc]/N1C1[index_2]
                B_[voc][0][index_1][index_2] = temp1 + temp2    
    
    return pi,A_1,B_1,A_,B_

def get_idx(mu):
    value = mu.max().max()
    for index_1 in voc_tag:
        for index_2 in voc_tag:
            if mu[index_1][index_2] == value: return (index_1,index_2)

def get_idx_value(mu):
    value = mu.max().max()
    for index_1 in voc_tag:
        for index_2 in voc_tag:
            if mu[index_1][index_2] == value: return index_1,value

def calculate(mu,A):
    temp = DataFrame(columns=voc_tag, index=range(1))
    result_mu = DataFrame(np.zeros([len(voc_tag),len(voc_tag)]),columns=voc_tag, index=voc_tag)
    result_phy = DataFrame(columns=voc_tag, index=voc_tag)
    for index in voc_tag:
        temp[index][0] = mu*A[index][0]
    for index_1 in voc_tag:
        for index_2 in voc_tag:
            max_v = -9999
            for index_3 in voc_tag:
                temp_value = temp[index_1][0][index_3][index_2]
                if temp_value > max_v:
                    max_v = temp_value
                    idx = index_3
            result_mu[index_2][index_1] = max_v
            result_phy[index_2][index_1] = idx
    return result_mu, result_phy


def Viterbi(pi,A,B,A_1,B_1,word):
    result = []
    if len(word) == 1:
        mu = DataFrame(np.zeros([len(voc_tag),len(word)]),columns=range(len(word)), index=voc_tag)
        idx = DataFrame(np.zeros([len(voc_tag),len(word)-1]),columns=range(len(word)-1), index=voc_tag)
        #initialization of first column
        mu.loc[:,0] = pi*B_1.iloc[list(B_1.index).index(word[0][0])]    
        return mu.idxmax(axis=0)
    mu = DataFrame(np.zeros([len(voc_tag),len(voc_tag)]),columns=voc_tag, index=voc_tag)
    
    mu = (A_1*B[word[1][0]][0]).apply(lambda x:x*(np.array(pi)*np.array(B_1.loc[word[0][0],:])),axis=1)
    if len(word) == 2:
        return get_idx(mu)
    phy = DataFrame(columns=range(len(word)-2), index=range(1))
    if len(word) > 2:
        for i in range(len(word)-2):
            
            result_mu, result_phy = calculate(mu,A)
            mu = result_mu*B[word[i+2][0]][0]
            phy[i][0] = result_phy
        index_last,index_llast=get_idx(mu)
        result.append(index_last)
        result.append(index_llast)
        for i in range(len(word)-2):
            result.insert(0,phy[len(word)-3-i][0][result[0]][result[1]])
    return result 


# Test function for certain trainset and testset.
# return baseline, result.
# for each result, as a list [error_rate, num_error_corrected, num_error_created, true_positive, false_negative]
def test_part(train,test):
    #caculate parameters of HMM according to train data
    pi,A_1,B_1,A_,B_ = calcule_Parametres(train)
    right_positive = 0.0
    positive = 0.0
    wrong_negative = 0.0
    negative = 0.0
    #test the performance for test set
    for word in test:
        corrected_word = Viterbi(pi,A_,B_,A_1,B_1,word)
        for i in range(len(word)):
            if word[i][0] == word[i][1]:
                positive += 1
                if corrected_word[i] == word[i][1]:
                    right_positive += 1
            else:
                negative += 1
                if not corrected_word[i] == word[i][1]:
                    wrong_negative += 1
    
    total = positive + negative
    baseline = negative / total
    
    res = []
    error_rate = (positive - right_positive + wrong_negative) / total
    res.append(error_rate)
    
    num_error_corrected = negative - wrong_negative
    res.append(num_error_corrected)
    
    num_error_created = positive - right_positive
    res.append(num_error_created)
    
    right_positive /= positive
    res.append(right_positive)
    
    wrong_negative /= negative
    res.append(wrong_negative)

    return baseline, res


if __name__=="__main__":
    baseline, viterbi = test_part(train10, test10)
    print("Here are the result of data-set with 10% error:")
    print("------------------------------------------------------")
    print("The baseline is: " + str(baseline))
    print("------------------------------------------------------")
    print("The error-rate using viterbi is: " + str(viterbi[0]))
    print("The number of corrected error using viterbi is: " + str(viterbi[1]))
    print("The number of created error using viterbi is: " + str(viterbi[2]))
    print("The true positive rate using viterbi is: " + str(viterbi[3]))
    print("The false negative rate using viterbi is: " + str(viterbi[4]))

    print("######################################################")

    baseline, viterbi = test_part(train20, test20)
    print("Here are the result of data-set with 20% error:")
    print("------------------------------------------------------")
    print("The baseline is: " + str(baseline))
    print("------------------------------------------------------")
    print("The error-rate using viterbi is: " + str(viterbi[0]))
    print("The number of corrected error using viterbi is: " + str(viterbi[1]))
    print("The number of created error using viterbi is: " + str(viterbi[2]))
    print("The true positive rate using viterbi is: " + str(viterbi[3]))
    print("The false negative rate using viterbi is: " + str(viterbi[4]))


