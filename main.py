import src.prep as prep 
import sys
import numpy as np 
from src.naive_bayes import naive_bayes
from src.non_naive_bayes import non_naive_bayes

def main():
    seed=12513
    if len(sys.argv)>1:
        seed=int(sys.argv[1])

    parr=1
    if len(sys.argv)>2:
        parr=int(sys.argv[2])
        parr=min(parr,21)

    alf=1.0
    if len(sys.argv)>3:
        alf=float(sys.argv[3])

    try:
        d_tr, res_tr, d_te, res_te, feat=prep.pre(seed)

        print("Testing accuracy: ")
        
        nb=naive_bayes(alf=alf)
        nb.fit(d_tr, res_tr)
        ans=nb.predict(d_te)
        ok=np.mean(ans==res_te)
        print(f"Naive Bayes: {100.0*ok}%")

        # daca dau mx_par prea mare o sa faca overfitting

        nnb=non_naive_bayes(alf=alf, mx_par=parr)
        nnb.fit(d_tr, res_tr)
        ans_nnb=nnb.predict(d_te)
        ok_nnb=np.mean(ans_nnb==res_te)
        print(f"Non-naive Bayes with mx_par={parr} {100.0*ok_nnb}%")

        print("Training accuracy: ")

        ans=nb.predict(d_tr)
        ok=np.mean(ans==res_tr)
        print(f"Naive Bayes: {100.0*ok}%")

        ans_nnb=nnb.predict(d_tr)
        ok_nnb=np.mean(ans_nnb==res_tr)
        print(f"Non-naive Bayes with mx_par={parr}: {100.0*ok_nnb}%")

    except Exception as e:  
        print(e)

if __name__=="__main__":
    main()