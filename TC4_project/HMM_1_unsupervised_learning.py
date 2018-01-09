from pandas import Series, DataFrame
import pandas as pd
import pickle
import numpy as np

# Address of dossier
adr_test10 = './data/test10.pkl'
adr_train10 = './data/train10.pkl'
adr_test20 = './data/test20.pkl'
adr_train20 = './data/train20.pkl'


# Read in the data
test10 = pickle.load(open(adr_test10, "rb"))
train10 = pickle.load(open(adr_train10, "rb"))
test20 = pickle.load(open(adr_test20, "rb"))
train20 = pickle.load(open(adr_train20, "rb"))
voc_tag = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
voc_dit = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}


# The implementation of forward-backward algrithm
# function to calculate parameters alpha
def forward_parameter(pi,A,B,word):
    alpha = DataFrame(np.zeros([len(voc_tag),len(word)]),columns=range(len(word)), index=voc_tag)
    #initialization of first column
    alpha.loc[:,0] = pi*B.iloc[list(B.index).index(word[0][0])]
    if len(word) == 1:
        return alpha
    else:
        for i in range(len(word)-1):
            alpha.loc[:,i+1] = B.iloc[list(B.index).index(word[i+1][0])]*((A*alpha.loc[:,i]).sum(axis=1))
        return alpha

# function to calculate parameters beta
def backward_parameter(pi,A,B,word):
    beta = DataFrame(np.zeros([len(voc_tag),len(word)]),columns=range(len(word)), index=voc_tag)
    #initialization of first column
    beta.loc[:,len(word)-1] = np.ones(len(voc_tag))
    if len(word) == 1:
        return beta
    else:
        for i in range(len(word)-1):
            beta.loc[:,len(word)-2-i] = (beta.loc[:,len(word)-1-i]*B.iloc[list(B.index).index(word[len(word)-1-i][0])]).dot(A)
        return beta
    
# compute the ideal word using forawrd-backward
def FB(pi,A,B,word):
    alpha = forward_parameter(pi,A,B,word)
    beta = backward_parameter(pi,A,B,word)
    joint_prob = alpha * beta
    return joint_prob.idxmax(axis=0)

# The implementation of viterbi algrithm
# compute the ideal word using viterbi
def VITERBI(pi,A,B,word):
    result = []
    mu = DataFrame(np.zeros([len(voc_tag),len(word)]),columns=range(len(word)), index=voc_tag)
    idx = DataFrame(np.zeros([len(voc_tag),len(word)-1]),columns=range(len(word)-1), index=voc_tag)
    #initialization of first column
    mu.loc[:,0] = pi*B.iloc[list(B.index).index(word[0][0])]
    if len(word) == 1:
        return mu.idxmax(axis=0)
    else:
        for i in range(len(word)-1):
            temp = A*mu.loc[:,i]
            idx.loc[:,i] = temp.idxmax(axis=1)
            mu.loc[:,i+1] = B.iloc[list(B.index).index(word[i+1][0])]*temp.max(axis=1)
        result.append((mu.loc[:,len(word)-1]).idxmax(axis=0))
        for i in range(len(word)-1):
            result.insert(0,idx.loc[result[0],len(word)-2-i]) 
        return result

def EM_learn(mydata):

    # We assume the value of observations and states are voc_tag
    # initialize A,B,pi
    A = DataFrame(np.random.rand(len(voc_tag), len(voc_tag)),columns=voc_tag, index=voc_tag)
    for index in voc_tag:
        A[index] /= float(sum(A[index]))
        
    B = DataFrame(np.random.rand(len(voc_tag), len(voc_tag)),columns=voc_tag, index=voc_tag)
    for index in voc_tag:
        B[index] /= float(sum(B[index]))
                  
    pi = np.random.rand(len(voc_tag))
    pi /= sum(pi)
    pi = Series(pi, index = voc_tag)
    
    #return forward_parameter(pi,A,B,mydata[0])
    num_epoch = 30
    for i in range(num_epoch):
        for word in mydata:
            alpha = forward_parameter(pi,A,B,mydata)
            beta = backward_parameter(pi,A,B,mydata)
            # gama = alpha * beta (dot product) / sum(alpha * beta) = p(y_t|X,lambda)
            gama = alpha * beta
            for c in gama.columns:
                gama[c] /= sum(gama[c])
            gama_tmoins1 = game.copy(deep)
            del gama_tmoins1[len(word)-1]
            # xi = p(y_t = i, y_t+1 = j |X,lambda) 26*26*(T-1)
            # xi is list which has 26 26*t DataFrames
            
            xi = np.zeros([len(voc_tag),len(voc_tag),len(word)-1])
            
            for i in range(len(voc_tag)):
                for j in range(len(voc_tag)):
                    for t in rang(len(word)-1):
                        xi[i][j][t] = gama[t][voc_dit[i]] * A[voc_dit[i]][voc_dit[j]] * B[voc_dit[j]][word[t+1][0]] \
                               * beta[t+1][voc_dit[j]] / beta[t][voc_dit[i]]
            # update A,B,pi
            pi = gama[0]
            for i in range(len(voc_tag)):
                for j in range(len(voc_tag)):
                    A[voc_dit[i]][voc_dit[j]] = xi[i][j].sum() / gama_tmoins1.loc[voc_dit[i]].sum()
            for i in range(len(voc_tag)):
                for j in range(len(voc_tag)):
                    tmp_sum = 0
                    for t in range(len(word)):
                        if word[t][0] == voc_dit[j]:
                            tmp_sum += game.loc[voc_dit[i]][t]
                    B[voc_dit[i]][voc_dit[j]] = tmp_sum / gama.loc[voc_dit[i]].sum()
                    
    return pi,A,B
    
# Test function for certain trainset and testset.
# return baseline, fb_result, viterbi_result.
# for each result, as a list [error_rate, num_error_corrected, num_error_created, true_positive, false_negative]
def test_part(train,test):
    #caculate parameters of HMM according to train data
    pi,A,B = calcule_parametres(train)
    right_positive_fb = 0.0
    right_positive_viterbi = 0.0
    positive = 0.0
    wrong_negative_fb = 0.0
    wrong_negative_viterbi = 0.0
    negative = 0.0
    #test the performance for test set
    for word in test:
        corrected_word_fb = FB(pi,A,B,word)
        corrected_word_viterbi = VITERBI(pi,A,B,word)
        for i in range(len(word)):
            if word[i][0] == word[i][1]:
                positive += 1
                if corrected_word_fb[i] == word[i][1]:
                    right_positive_fb += 1
                if corrected_word_viterbi[i] == word[i][1]:
                    right_positive_viterbi += 1
            else:
                negative += 1
                if not corrected_word_fb[i] == word[i][1]:
                    wrong_negative_fb += 1
                if not corrected_word_viterbi[i] == word[i][1]:
                    wrong_negative_viterbi += 1
    
    # compute several evaluation critics
    total = positive + negative
    baseline = negative / total
    
    fb = []
    error_rate = (positive - right_positive_fb + wrong_negative_fb) / total
    fb.append(error_rate)
    
    num_error_corrected = negative - wrong_negative_fb
    fb.append(num_error_corrected)
    
    num_error_created = positive - right_positive_fb
    fb.append(num_error_created)
    
    right_positive_fb /= positive
    fb.append(right_positive_fb)
    
    wrong_negative_fb /= negative
    fb.append(wrong_negative_fb)
    
    viterbi = []
    error_rate = (positive - right_positive_viterbi + wrong_negative_viterbi) / total
    viterbi.append(error_rate)
    
    num_error_corrected = negative - wrong_negative_viterbi
    viterbi.append(num_error_corrected)
    
    num_error_created = positive - right_positive_viterbi
    viterbi.append(num_error_created)
    
    right_positive_viterbi /= positive
    viterbi.append(right_positive_viterbi)
    wrong_negative_viterbi /= negative
    viterbi.append(wrong_negative_viterbi)
    
    return baseline, fb, viterbi


if __name__=="__main__":
	baseline, fb, viterbi = test_part(train10, test10)
	print("Here are the result of data-set with 10% error:")
	print("------------------------------------------------------")
	print("The baseline is: " + str(baseline))
	print("------------------------------------------------------")
	print("The error-rate using fb is: " + str(fb[0]))
	print("The number of corrected error using fb is: " + str(fb[1]))
	print("The number of created error using fb is: " + str(fb[2]))
	print("The true positive rate using fb is: " + str(fb[3]))
	print("The false negative rate using fb is: " + str(fb[4]))
	print("------------------------------------------------------")
	print("The error-rate using viterbi is: " + str(viterbi[0]))
	print("The number of corrected error using viterbi is: " + str(viterbi[1]))
	print("The number of created error using viterbi is: " + str(viterbi[2]))
	print("The true positive rate using viterbi is: " + str(viterbi[3]))
	print("The false negative rate using viterbi is: " + str(viterbi[4]))

	print("######################################################")

	baseline, fb, viterbi = test_part(train20, test20)
	print("Here are the result of data-set with 20% error:")
	print("------------------------------------------------------")
	print("The baseline is: " + str(baseline))
	print("------------------------------------------------------")
	print("The error-rate using fb is: " + str(fb[0]))
	print("The number of corrected error using fb is: " + str(fb[1]))
	print("The number of created error using fb is: " + str(fb[2]))
	print("The true positive rate using fb is: " + str(fb[3]))
	print("The false negative rate using fb is: " + str(fb[4]))
	print("------------------------------------------------------")
	print("The error-rate using viterbi is: " + str(viterbi[0]))
	print("The number of corrected error using viterbi is: " + str(viterbi[1]))
	print("The number of created error using viterbi is: " + str(viterbi[2]))
	print("The true positive rate using viterbi is: " + str(viterbi[3]))
	print("The false negative rate using viterbi is: " + str(viterbi[4]))