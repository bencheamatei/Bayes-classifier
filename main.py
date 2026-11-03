import src.prep as prep 

def main():
    d_tr, res_tr, d_te, res_te, feat=prep.pre(67)
    print(feat)

if __name__=="__main__":
    main()