import torch
import pickle
import numpy as np
from tqdm import tqdm
from model.GCN_E import GCN
from model.GCN_ddec import SemiFullGN
from model.data_ddec import collate_pool, get_data_loader, CIFData

def load_gcn(gcn_name):
    checkpoint = torch.load(gcn_name)
    x = checkpoint['model_args']
    atom_fea_len = x['atom_fea_len']
    h_fea_len = x['h_fea_len']
    n_conv = x['n_conv']
    n_h = x['n_h']
    orig_atom_fea_len = x['orig_atom_fea_len']
    nbr_fea_len = x['nbr_fea_len']
    model =GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def main():
    model_folder = 'saved_models'
    best_name = model_folder+'/'+'best_ddec_re/ddec.pth'
    best_gcn = model_folder+'/'+'best_pbe_more/pbe-atom.pth'
    gcn = load_gcn(best_gcn)

    root_dir ='./data/json/'
    root_dir_pos ='./data/npy/pos/'
    root_dir_cell ='./data/npy/cell/'
    root_dir_ddec ='./data/npy/ddec/'
    radius = 6
    dmin = 0
    step = 0.2
    random_seed = 1126
    batch_size = 1
    num_workers = 0
    pin_memory = False
    pre_csv = 'list-ddec.csv'
    pre_dataset = CIFData(root_dir,root_dir_pos,root_dir_cell,root_dir_ddec,pre_csv,radius,dmin,step,random_seed)
    collate_fn = collate_pool
    pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
    with open(model_folder + '/best_ddec/normalizer-ddec.pkl', 'rb') as f:
        normalizer = pickle.load(f)

    structures, _,_,_ = pre_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1] + 3
    nbr_fea_len = structures[1].shape[-1]
    
    atom_fea_len = 128
    h_fea_len = 256
    n_conv = 8
    
    checkpoint = torch.load(best_name)
    model = SemiFullGN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
   
    ddec_save_folder = './predict/charge/'
    for _, (input,_,cif_ids) in enumerate(tqdm(pre_loader)):
        with torch.no_grad():
            input_var = (input[0].cuda(),
						input[1].cuda(),
						input[2].cuda(),
						input[3].cuda(),
						input[4].cuda(),
						input[5].cuda())
            relaxed_feature = gcn.Encoding(*input_var)
            atoms_fea = torch.cat((input[0],input[7]),dim=-1)
            input_var2 = (atoms_fea.cuda(),
                        input[1].cuda(),
                        input[2].cuda(),
                        input[3].cuda(),
                        input[4].cuda(),
                        input[5].cuda(),
                        relaxed_feature,
                        input[9][:,:9].cuda()) 
            output = model(*input_var2)
            output = normalizer.denorm(output.data.cpu())
            name = cif_ids[0]+'.npy'
            np.save(ddec_save_folder+name,output)

if __name__ == '__main__':
    main()
