import torch
import pickle
import pandas as pd
from tqdm import tqdm
from model.GCN_E import GCN
from torch.autograd import Variable
from model.data_pbe import collate_pool, get_train_val_test_loader, CIFData

def main():
    model_folder = 'model/'
    model_pbe = model_folder+'best_pbe/pbe-atom.pth'
    # model_bandgap = model_folder+'best_bandgap/bandgap.pth'
    root_dir = './data/json/'
    radius = 6.0
    dmin = 0
    step = 0.2
    random_seed = 1123
    batch_size = 1
    N_tot = 16781
    atom_fea_len = 128
    h_fea_len = 256
    n_conv = 7
    n_h = 5
    N_tr = int(N_tot*0.8)
    N_val = int(N_tot*0.1)
    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tot))
    pin_memory = False
    num_workers = 0
    dataset = CIFData(root_dir,radius,dmin,step,random_seed=random_seed)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset,collate_fn,batch_size,train_idx,val_idx,test_idx,num_workers,pin_memory)
    # sample_target = sampling(root_dir+'/'+'id_prop_relaxed.csv')
    # normalizer = Normalizer(sample_target)
    with open(model_folder + '/best_bandgap/normalizer-bandgap.pkl', 'rb') as f:
        normalizer = pickle.load(f)
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    checkpoint = torch.load(model_pbe)
    # checkpoint = torch.load(model_bandgap)
    x = checkpoint['model_args']
    atom_fea_len = x['atom_fea_len']
    h_fea_len = x['h_fea_len']
    n_conv = x['n_conv']
    n_h = x['n_h']
    model = GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    ## train
    pbe_results_train = []
    for _, (input,target,cifids) in enumerate(tqdm(train_loader)):
        input_var = (Variable(input[0].cuda()),
                            input[1].cuda(),
                            input[2].cuda(),
                            input[3].cuda(),
                            input[4].cuda(),
                            input[5].cuda())
        output = model(*input_var)
        pbe_results_train.append([cifids[0],target.item(),normalizer.denorm(output.data.cpu()).item()])
    df_pbe_results_train = pd.DataFrame(pbe_results_train, columns=["name","DFT-per","GCN-per"])
    n_atom = pd.read_csv("./predict/pbe/atom_number.csv")
    p_train = []
    for name in df_pbe_results_train["name"]:
        p_d = df_pbe_results_train[df_pbe_results_train["name"]==name]["DFT-per"].values.ravel()
        p_p = df_pbe_results_train[df_pbe_results_train["name"]==name]["GCN-per"].values.ravel()
        n = n_atom[n_atom["name"]==name]["num"].tolist()[0]
        p_d_a = p_d*n
        p_p_a = p_p*n
        p_train.append([float(p_d_a),float(p_p_a)])
    df_p_train = pd.DataFrame(p_train,columns = ["DFT-all","GCN-all"])
    df_data_train = pd.concat([df_pbe_results_train, df_p_train], axis=1)
    df_data_train.to_csv("./predict/pbe/predicted_pbe_train.csv",index=False)
    # df_data_train.to_csv("./predict/bandgap/predicted_bandgap_train.csv",index=False)
    print("training set:")
    print(df_data_train)

    ## val
    pbe_results_val = []
    for _, (input,target,cifids) in enumerate(tqdm(val_loader)):
        input_var = (Variable(input[0].cuda()),
                            input[1].cuda(),
                            input[2].cuda(),
                            input[3].cuda(),
                            input[4].cuda(),
                            input[5].cuda())
        output = model(*input_var)
        pbe_results_val.append([cifids[0],target.item(),normalizer.denorm(output.data.cpu()).item()])
    df_pbe_results_val = pd.DataFrame(pbe_results_val, columns=["name","DFT-per","GCN-per"])
    n_atom = pd.read_csv("./predict/pbe/atom_number.csv")
    p_val = []
    for name in df_pbe_results_val["name"]:
        p_d = df_pbe_results_val[df_pbe_results_val["name"]==name]["DFT-per"].values.ravel()
        p_p = df_pbe_results_val[df_pbe_results_val["name"]==name]["GCN-per"].values.ravel()
        n = n_atom[n_atom["name"]==name]["num"].tolist()[0]
        p_d_a = p_d*n
        p_p_a = p_p*n
        p_val.append([float(p_d_a),float(p_p_a)])
    df_p_val = pd.DataFrame(p_val,columns = ["DFT-all","GCN-all"])
    df_data_val= pd.concat([df_pbe_results_val, df_p_val], axis=1)
    df_data_val.to_csv("./predict/pbe/predicted_pbe_val.csv",index=False)
    # df_data_val.to_csv("./predict/bandgap/predicted_bandgap_val.csv",index=False)
    print("validation set:")
    print(df_data_val)

    ## test
    pbe_results_test = []
    for _, (input,target,cifids) in enumerate(tqdm(test_loader)):
        input_var = (Variable(input[0].cuda()),
                            input[1].cuda(),
                            input[2].cuda(),
                            input[3].cuda(),
                            input[4].cuda(),
                            input[5].cuda())
        output = model(*input_var)
        pbe_results_test.append([cifids[0],target.item(),normalizer.denorm(output.data.cpu()).item()])
    df_pbe_results_test = pd.DataFrame(pbe_results_test, columns=["name","DFT-per","GCN-per"])
    n_atom = pd.read_csv("./predict/pbe/atom_number.csv")
    p_test = []
    for name in df_pbe_results_test["name"]:
        p_d = df_pbe_results_test[df_pbe_results_test["name"]==name]["DFT-per"].values.ravel()
        p_p = df_pbe_results_test[df_pbe_results_test["name"]==name]["GCN-per"].values.ravel()
        n = n_atom[n_atom["name"]==name]["num"].tolist()[0]
        p_d_a = p_d*n
        p_p_a = p_p*n
        p_test.append([float(p_d_a),float(p_p_a)])
    df_p_test = pd.DataFrame(p_test,columns = ["DFT-all","GCN-all"])
    df_data_test = pd.concat([df_pbe_results_test, df_p_test], axis=1)
    df_data_test.to_csv("./predict/pbe/predicted_pbe_test.csv",index=False)
    # df_data_test.to_csv("./predict/bandgap/predicted_bandgap_test.csv",index=False)
    print("test set:")
    print(df_data_test)

if __name__ == '__main__':
    main()

