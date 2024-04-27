<h1 align="center">PACMAN</h1>

<h4 align="center">

</h4>              

**P**artial **A**tomic **C**harges for Porous **Ma**terials based on Graph Convolutional Neural **N**etwork (**PACMAN**)                           

[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)[![PyPI version](https://badge.fury.io/py/pyEQL.svg)](https://pypi.org/project/GCNCharge/) [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10822403-blue)](https://doi.org/10.5281/zenodo.10822403)  [![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sxm13/GCNCharges/LICENSE.txt) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sxmzhaogb@gmail.com) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)]() [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)]()          
![Logo](./figs/toc.png)                      



# Installation                             
                                 
:star: **by [pip](https://pypi.org/project/GCNCharge/)**                                                              

```sh
pip install GCNCharge
```

## by source                                                                  

**Download**                          

```sh
git clone https://github.com/sxm13/GCNCharges.git
```   
                               
```sh
pip install -r requirements.txt
```

# Charge Assignment               
You can put your cif files in any folder, but please run the code and jupyter notebook in this folder.                

**bash**
```sh
python pacman.py [folder name] [MOF/COF] [digits]
```
example: ```python pacman.py test_file MOF 10```

:star: **notebook(from pip)**                      
                
```sh      
from GCNCharge import GCNCharge
GCNCharge.predict(cif_file="Cu-BTC.cif",model_name="MOF",di=10,neutral=True)
```

file: your folder contains cif files                               
model: MOF or COF                                                   

# Website & Zenodo
*  IF you do not want to install PACMAN, you can go to this :point_right: [link](https://pacman-mtap.streamlit.app/).       
*  IF you want to DOWNLOAD full code and dataset, you can go to this :point_right: [link](https://zenodo.org/records/10822403) But we will not update new vesion in Zenodo, new vesion will upload here.            

# Reference
If you use GCN Charge, please cite [this paper]():
```bib
@article{,
    title={A Robust Partial Atomic Charge Estimator for Nanoporous Materials using Crystal Graph Convolution Network},
    DOI={},
    journal={Journal of Chemical Theory and Computation},
    author={Zhao, Guobin and Chung, Yongchul},
    year={2024},
    pages={}
}
```

# Bugs

 If you encounter any problem during using ***PACMAN***, please talk to me ```sxmzhaogb@gmail.com```.                   

 
# Development

                  
| Database with DDEC Charges                                                                                                                                      | url                                                                                                                                        | size                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QMOF | [link](https://github.com/Andrew-S-Rosen/QMOF) | 16,779 |
| CoRE MOF 2014 DDEC | [link](https://zenodo.org/records/3986573#.XzfKiJMzY8N) | 2,932 |
| CoRE MOF 2014 DFT-optimized | [link](https://zenodo.org/records/3986569#.XzfKcpMzY8N) | 502 | 
| CURATED-COFs | [link](https://github.com/danieleongari/CURATED-COFs) | 612 |

#### Workflow            
<img src="./figs/workflow.png" alt="workflow" width="500">             
                    
### Folder explain
```
.
â”œâ”€â”? ..
â”œâ”€â”? figs                                                # Figures used for introduction 
â”?   â”œâ”€â”? toc.jpg                                         # Table of Contents
â”?   â””â”€â”? workflow.png                                    # Workflow of this project
â”?
â”œâ”€â”? model                                               # Python files used for dataset prepartion & GCN training
â”?   â”œâ”€â”? GCN_E.py                                        # Networks model for energy/bandgap training
â”?   â”œâ”€â”? GCN_ddec.py                                     # Networks model for atomic charge training
â”?   â”œâ”€â”? cif2data.py                                     # Convert QMOF database to dataset
â”?   â”œâ”€â”? data_E.py                                       # Convert cif to graph & target (energy/bandgap)
â”?   â”œâ”€â”? data_ddec.py                                    # Convert cif to graph & target (atomic charge)
â”?   â””â”€â”? utils.py                                        # Normalizer, sampling, AverageMeter, save_checkpoint
â”?
â”œâ”€â”? model4pre                                           # Python files used for prediction
â”?   â”œâ”€â”? GCN_E.py                                        # Networks model for energy/bandgap prediction
â”?   â”œâ”€â”? GCN_ddec.py                                     # Networks model for atomic charge prediction
â”?   â”œâ”€â”? atom_init.json                                  # 
â”?   â”œâ”€â”? cif2data.py                                     # Read/write cif file
â”?   â”œâ”€â”? data.py                                         # Convert cif to graph & target (energy/bandgap)
â”?   â”œâ”€â”? data_ddec.py                                    # Convert cif to graph & target (atomic charge)
â”?   â””â”€â”? utils.py                                        # Normalizer, sampling, AverageMeter, save_checkpoint
â”?
â”œâ”€â”? pth                                                 # Models of this project
â”?   â”œâ”€â”? best_bandgap                                    # Bandgap
â”?   â”?   â”œâ”€â”? bandgap.pth                                 # Bandgap model
â”?   â”?   â””â”€â”? normalizer-bandgap.pkl                      # Normalizer of bandgap
â”?   â”œâ”€â”? best_ddec                                       # MOF DDEC
â”?   â”?   â”œâ”€â”? ddec.pth                                    # ///
â”?   â”?   â””â”€â”? normalizer-ddec.pkl                         # ///
â”?   â”œâ”€â”? best_ddec_COF                                   # ///
â”?   â”?   â”œâ”€â”? ddec.pth                                    # ///
â”?   â”?   â””â”€â”? normalizer-ddec.pkl                         # ///
â”?   â”œâ”€â”? best_pbe                                        # ///
â”?   â”?   â”œâ”€â”? pbe-atom.pth                                # ///
â”?   â”?   â””â”€â”? normalizer-pbe.pkl                          # ///
â”?   â”œâ”€â”? chk_bandgap                                     # Bandgap
â”?   â”?   â””â”€â”? checkpoint.pth                              # Checkpoint of bandgap
â”?   â”œâ”€â”? chk_ddec                                        # ///
â”?   â”?   â””â”€â”? checkpoint.pth                              # ///
â”?   â””â”€â”? chk_pbe                                         # ///
â”?       â””â”€â”? checkpoint.pth                              # ///
â”?
â”œâ”€â”? GCNCharge.ipynb                                     # notebook example for atomic charge assignment
â”œâ”€â”? GCNCharge.py                                        # main python file for atomic charge assignment by command line
â”œâ”€â”? GCNCharge4notebook.py                               # main python file for atomic charge assignment by notebook
â”œâ”€â”? LICENSE.txt                                         # MIT license
â”œâ”€â”? README.md                                           # Usage/Source
â”œâ”€â”? predict_E.py                                        # main python file for energy/bandgap prediction
â”œâ”€â”? predict_ddec.py                                     # main python file for atomic charge prediction
â”œâ”€â”? requirements.txt                                    # packages need to be installed
â”œâ”€â”? train_E.py                                          # main python file for energy/bandgap training
â””â”€â”? train_ddec.py                                       # main python file for atomic charge training

```

 
**Group:**   [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home?authuser=0)                                
