import numpy as np 

class non_naive_bayes:
    def __init__(self, alf=1.0, corr=0.01, mx_par=1):
        self.alf=alf
        self.corr=corr
        self.mx_par=mx_par  
        self.prior=None 
        self.ps=None 
        self.dep=None # dependentele 

    def calc_dependencies(self, x):
        _, m=x.shape
        curr=np.abs(np.corrcoef(x, rowvar=False))
        self.dep={}

        for col in range(m):
            ant=curr[col, :col]
            aux=[i for i in range(col) if ant[i]>=self.corr]
            aux.sort(key=lambda i: ant[i], reverse=True)
            self.dep[col]=aux[:min(len(aux), self.mx_par)]

        return self 

    def fit(self, x, y):
        n, m=x.shape
        cnt1=np.sum(y)
        self.prior=[(n-cnt1)/n, cnt1/n]
        self.calc_dependencies(x)

        self.ps=[[[{} for _ in range(m)] for _ in range(2)] for _ in range(2)]
        y=y.astype(np.int64)

        for col in range(m):
            k=len(self.dep[col])
            code=np.zeros(n, dtype=np.int64)
            for bit in range(k):
                code|=x[:,self.dep[col][bit]].astype(np.int64)<<bit

            aux=y*(1<<k)+code
            tot=np.bincount(aux, minlength=2*(1<<k)).reshape(2, (1<<k))
            cnt=np.bincount(aux[x[:,col]==1], minlength=2*(1<<k)).reshape(2, (1<<k))
            p1=(cnt+self.alf)/(tot+2*self.alf)

            for rez in range(2):
                for mask in range(1<<k):
                    self.ps[1][rez][col][mask]=p1[rez, mask]
                    self.ps[0][rez][col][mask]=1-p1[rez, mask]
        return self

    def calc_logs(self, x):
        s=[np.log(self.prior[0]), np.log(self.prior[1])]
        for rez in range(2):
            for col in range(len(x)):
                mask=0
                for i in range(len(self.dep[col])):
                    if x[self.dep[col][i]]==1:
                        mask|=(1<<i)
                s[rez]+=np.log(self.ps[x[col]][rez][col][mask])
        return s[0], s[1]

    def predict(self, x):
        def predict_row(x):
            s0, s1=self.calc_logs(x)
            if s0>=s1:
                return 0
            return 1
        return np.array([predict_row(row) for row in x])