import src.prep as prep 
import sys
import numpy as np 
from src.non_naive_bayes import non_naive_bayes
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():

    if len(sys.argv)<2:
        print("da nr computatiei\n")
        exit(0)

    try:
        if int(sys.argv[1])==1:
            nr_itr=5
            test_acc=[0.0 for i in range(22)]
            train_acc=[0.0 for i in range(22)]
            for _ in range(nr_itr):
                seed=np.random.randint(1, 1000000)
                d_tr, res_tr, d_te, res_te, feat=prep.pre(seed)

                for max_par in range(22):
                    model=non_naive_bayes(mx_par=max_par)
                    model.fit(d_tr, res_tr)

                    ans_train=model.predict(d_tr)
                    ans_test=model.predict(d_te)

                    ok_train=np.mean(ans_train==res_tr)
                    ok_test=np.mean(ans_test==res_te)

                    test_acc[max_par]+=ok_test/nr_itr
                    train_acc[max_par]+=ok_train/nr_itr

            for i in range(22):
                print(f"({i}, {train_acc[i]:.2%})")
            print("---")
            for i in range(22):
                print(f"({i}, {test_acc[i]:.2%})")
        else:
            nr_itr=50
            models=["Naiv", "Nenaiv (mx=4)", "Random Forest", "XGBoost"]
            acc_test={m: 0.0 for m in models}
            for i in range(nr_itr):
                seed=np.random.randint(1, 1000000)
                d_tr, res_tr, d_te, res_te, feat = prep.pre(seed)
                
                nb=non_naive_bayes(mx_par=0).fit(d_tr, res_tr)
                acc_test["Naiv"]+=np.mean(nb.predict(d_te) == res_te)/nr_itr
                
                nnb=non_naive_bayes(mx_par=4).fit(d_tr, res_tr)
                acc_test["Nenaiv (mx=4)"]+=np.mean(nnb.predict(d_te)==res_te)/nr_itr

                d_tr, res_tr, d_te, res_te, feat = prep.pre(seed, raw=True)
                
                rf=RandomForestClassifier(random_state=seed, n_jobs=-1)
                rf.fit(d_tr, res_tr)
                acc_test["Random Forest"]+=np.mean(rf.predict(d_te) == res_te) / nr_itr
                
                xgb=XGBClassifier(random_state=seed, eval_metric='logloss', n_jobs=-1)
                xgb.fit(d_tr, res_tr)
                acc_test["XGBoost"]+=np.mean(xgb.predict(d_te) == res_te) / nr_itr
            for model in models:
                print(f"{model}: {acc_test[model]:.2%}")
    except Exception as e:  
        print(e)

if __name__=="__main__":
    main()