import numpy as np

class naive_bayes:

    # atunci cand calculam P(Xi = 0/1 | Y=0/1)
    # e posibil ca una dintre probabilitatile noastre sa dea 0
    # lucru care ar face intregul produs egal cu 0, ceea ce nu e bine
    # ca sa scap de chestia asta exista ceea ce se numeste laplace smoothing 
    # care in esenta imi ia x/y si il face (x+a)/(y+cnt*a), unde 
    # cnt vine nr de clase posibile din clasificare, deci in cazul meu fix 2 
    def __init__(self, alf=1.0):
        self.alf=alf
        self.prior=None # aici o sa am sansele de diabet/non diabet initiale
        
        self.ps=None # o sa fie o tabela de tipul ps[0/1][0/1][nr]
                     # care tine P(X_nr=0/1|Y=0/1)

    def fit(self, x, y):
        n, m=x.shape        
        cnt1=np.sum(y)
        self.prior=[(n-cnt1)/n, cnt1/n]

        self.ps=[[[0.0 for _ in range(m)] 
                  for _ in range(2)] for _ in range(2)]
        
        for rez in range(2):
            tot=np.sum(y==rez)
            for col in range(m):
                cnt=np.sum(((y==rez)&(x[:,col]==1)))
                self.ps[1][rez][col]=(cnt+self.alf)/(tot+2*self.alf)
                self.ps[0][rez][col]=1.0-self.ps[1][rez][col]

        return self

    def calc_logs(self, x):
        s=[np.log(self.prior[0]), np.log(self.prior[1])]
        for rez in range(2):
            for col in range(len(x)):
                s[rez]+=np.log(self.ps[x[col]][rez][col])
        return s[0], s[1]

    def predict(self, x):
        def predict_row(x):
            s0, s1=self.calc_logs(x)
            if s0>=s1:
                return 0
            return 1
        return np.array([predict_row(row) for row in x])