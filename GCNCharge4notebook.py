import sys
import os
import glob
import json
import torch
import pickle
import sys
import importlib
import numpy as np
from tqdm import tqdm
from model4pre.GCN_ddec import SemiFullGN
from model4pre.data import collate_pool, get_data_loader, CIFData, load_gcn
from model4pre.cif2data import ase_format, CIF2json, pre4pre, write4cif   #,n_atom

def GCNChagre(file,model,digits):

    source = importlib.import_module('model4pre')
    sys.modules['source'] = source
    # if int(digits)<6:
    print("Digits Warning: please note that this model is trained on 6 decimal places.")
    model_type = model
    model_name = "COF" if model_type == "COF" else "MOF"
    print(f"model name: {model_name}")
    path = file
    # folder_name = path
    if os.path.isfile(path):
        print("please input a folder, not a file")
    elif os.path.isdir(path):
        pass
    else:
        print("Can not find your file, please check is it exit or correct?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_pbe_name = "./pth/best_pbe/pbe-atom.pth"
    if model_type == "COF":
        model_ddec_name = "./pth/best_ddec_COF/ddec.pth"
        ddec_nor_name = "./pth/best_ddec_COF/normalizer-ddec.pkl"
    else:
        model_ddec_name = "./pth/best_ddec/ddec.pth"
        ddec_nor_name = "./pth/best_ddec/normalizer-ddec.pkl"
    gcn = load_gcn(model_pbe_name)
    with open(ddec_nor_name, 'rb') as f:
        ddec_nor = pickle.load(f)
    f.close()

    all_cif_files = glob.glob(os.path.join(path, '*.cif'))
    cif_files = [f for f in all_cif_files if not f.endswith('_gcn.cif')]
    # dic = {}
    fail = {}
    i = 0
    for path in tqdm(cif_files):
        try:
            ase_format(path)
            CIF2json(path,save_path="")
            pre4pre(path,"","")
            # num_atom = n_atom(path)
            batch_size = 1
            num_workers = 0
            pin_memory = False
            pre_dataset = CIFData(path,"","")
            collate_fn = collate_pool
            pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
            structures, _, _ = pre_dataset[0]
            chg_1 = structures[0].shape[-1] + 3
            chg_2 = structures[1].shape[-1]
            chkpt_ddec = torch.load(model_ddec_name, map_location=torch.device(device))
            model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
            model4chg.cuda() if torch.cuda.is_available() else model4chg.to(device)
            model4chg.load_state_dict(chkpt_ddec['state_dict'])
            model4chg.eval()
            for _, (input,cif_ids) in enumerate(pre_loader):
                with torch.no_grad():
                    if device == "cuda":
                        input_cuda = [input_tensor.to(device) for input_tensor in input]
                        input_var = (input_cuda[0].cuda(),
                                    input_cuda[1].cuda(),
                                    input_cuda[2].cuda(),
                                    input_cuda[3].cuda(),
                                    input_cuda[4].cuda(),
                                    input_cuda[5].cuda())
                        encoder_feature = gcn.Encoding(*input_var)
                        atoms_fea = torch.cat((input_cuda[0],input_cuda[7]),dim=-1)
                        input_var2 = (atoms_fea.cuda(),
                                        input_cuda[1].cuda(),
                                        input_cuda[2].cuda(),
                                        input_cuda[3].cuda(),
                                        input_cuda[4].cuda(),
                                        input_cuda[5].cuda(),
                                        encoder_feature.cuda(),
                                        input_cuda[9][:,:9].cuda())
                    else:
                        input_var = (input[0],
                                    input[1],
                                    input[2],
                                    input[3],
                                    input[4],
                                    input[5])
                        encoder_feature = gcn.Encoding(*input_var)
                        atoms_fea = torch.cat((input[0],input[7]),dim=-1)
                        input_var2 = (atoms_fea,
                                        input[1],
                                        input[2],
                                        input[3],
                                        input[4],
                                        input[5],
                                        encoder_feature,
                                        input[9][:,:9])
                    chg = model4chg(*input_var2)
                    chg = ddec_nor.denorm(chg.data.cpu())
                    name = cif_ids[0]+'_charge.npy'
                    np.save(""+name,chg)
                    write4cif(path,"","",digits,charge = True)
                    print("writing cif: " + cif_ids[0] + "_gcn.cif")
                    os.remove(cif_ids[0] + '.json')
                    os.remove(cif_ids[0] + '_cell.npy')
                    os.remove(cif_ids[0] + '_pos.npy')
                    os.remove(cif_ids[0] + '_charge.npy')
                    i+=1
        except:
            print("Fail predict: " + path)
        #     fail[str(i)]=[path]
        #     i += 1
        # with open(folder_name + "fail.json",'w') as f:
        #     json.dump(fail,f)
        # f.close()
