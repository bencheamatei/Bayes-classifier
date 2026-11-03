import src.prep as prep 
import sys
import numpy as np 
from src.naive_bayes import naive_bayes
from src.non_naive_bayes import non_naive_bayes

def main():
    seed=12513
    if len(sys.argv)>1:
        seed=int(sys.argv[1])
    try:
        d_tr, res_tr, d_te, res_te, feat=prep.pre(seed)

        nb=naive_bayes(alf=1.0)
        nb.fit(d_tr, res_tr)
        ans=nb.predict(d_te)
        ok=np.mean(ans==res_te)
        print(f"{100.0*ok}%")

        # daca dau mx_par prea mare o sa faca overfitting
        nnb=non_naive_bayes(alf=1.0, mx_par=3)
        nnb.fit(d_tr, res_tr)
        ans_nnb=nnb.predict(d_te)
        ok_nnb=np.mean(ans_nnb==res_te)
        print(f"{100.0*ok_nnb}%")

    except Exception as e:  
        print(e)

if __name__=="__main__":
    main()