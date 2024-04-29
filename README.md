<h1 align="center">PACMAN</h1>

<h4 align="center">

</h4>              

A **P**artial **A**tomic **C**harge Predicter for Porous **Ma**terials based on Graph Convolutional Neural **N**etwork (**PACMAN**).

- DDEC6, Bader, CM5 for metal-organic frameworks (MOFs)
- DDEC6 for covalent-organic frameworks (COFs)

[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)[![PyPI version](https://badge.fury.io/py/pyEQL.svg)](https://pypi.org/project/PACMANCharge/) [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10822403-blue)](https://doi.org/10.5281/zenodo.10822403)  [![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sxm13/PACMAN/LICENSE.txt) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sxmzhaogb@gmail.com) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)]() [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)]()          
                     

# Installation                             
                                 
:star: **by [pip](https://pypi.org/project/PACMANCharge/)**                                                              

```sh
pip install PACMAN-Charge
```

## by source                                                                  

**Download**                          

```sh
git clone https://github.com/sxm13/PACMAN.git
cd PACMAN
pip install -r requirements.txt
```                            
         
# Charge Assignment               
           
:star: **notebook(from pip)**                      
                
```sh      
from PACMANCharge import pmcharge
PACMaN.predict(cif_file="./test/Cu-BTC.cif",model_name="MOF",charge_type="DDEC6",digits=10,atom_type=True,neutral=True)

```

cif_file: cif file  
                                                                              
                                  
**bash**
```sh
python PACMaN.py folder-name[path] model-name[MOF/COF] charge-type[DDEC6/Bader/CM5] digits[int] atom-type[True/False] neutral[True/False]
```
example: ```python PACMaN.py test_file/test-1/ MOF DDEC6 10 True True```

* folder-name: folder with cif files (without partial atomic charges).                               
* model_name & model-name: MOF or COF(COF just can use DDEC6)   
* charge-type: Charge type, DDEC6, Bader or CM5.             
* digits: digits of charge (recommond use 6). ML models were trained on 6 digit dataset.                                                       
* atom_type & atom-type: keep the same partial atomic charge for the same atom types (based on the similarity of partial atomic charges).                                     
* neutral: keep the net charge is zero. We use "mean" method to neuralize the system. Please refer to the manuscript about the method.                     

# Website & Zenodo
* You can predict partial atomic charges using an online APP :point_right: [link](https://pacman-charge-mtap.streamlit.app/).       
* Full code and dataset can be downloaded from :point_right: [link](https://zenodo.org/records/10822403)
* Note: All future releases will be uploaded on Github and pip only.

# Reference
If you use PACMAN Charge, please cite [this paper]():
```bib
@article{,
    title={PACMAN: A Robust Partial Atomic Charge Predicter for Nanoporous Materials using Crystal Graph Convolution Network},
    DOI={},
    journal={Journal of Chemical Theory and Computation},
    author={Zhao, Guobin and Chung, Yongchul},
    year={2024},
    pages={}
}
```

# Bugs

 If you encounter any problem during using ***PACMAN***, please email ```sxmzhaogb@gmail.com``` or create "issues".

 
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
├── ..
├── figs                                                # Figures used for introduction 
│   ├── toc.jpg                                         # Table of Contents
│   ├── workflow.png                                    # Workflow of this project
│
├── model                                               # Python files used for dataset prepartion & GCN training
│   ├── GCN_E.py                                        # Networks model for energy/bandgap training
│   ├── GCN_ddec.py                                     # Networks model for atomic charge training
│   ├── cif2data.py                                     # Convert QMOF database to dataset
│   ├── data_E.py                                       # Convert cif to graph & target (energy/bandgap)
│   ├── data_ddec.py                                    # Convert cif to graph & target (atomic charge)
│   └── utils.py                                        # Normalizer, sampling, AverageMeter, save_checkpoint
│
├── model4pre                                           # Python files used for prediction
│   ├── GCN_E.py                                        # Networks model for energy/bandgap prediction
│   ├── GCN_ddec.py                                     # Networks model for atomic charge prediction
│   ├── atom_init.json                                  # 
│   ├── cif2data.py                                     # Read/write cif file
│   ├── data.py                                         # Convert cif to graph & target (energy/bandgap)
│   ├── data_ddec.py                                    # Convert cif to graph & target (atomic charge)
│   └── utils.py                                        # Normalizer, sampling, AverageMeter, save_checkpoint
│
├── pth                                                 # Models of this project
    ├── best_bader                                      # Bader
│   │   ├── bader  .pth                                 # Bader charge model
│   │   └── normalizer-bader.pkl                        # Normalizer of bandgap
│   ├── best_bandgap                                    # Bandgap
│   │   ├── bandgap.pth                                 # Bandgap model
│   │   └── normalizer-bandgap.pkl                      # Normalizer of bandgap
    ├── best_cm5                                        # CM5
│   │   ├── bandgap.pth                                 # ///
│   │   └── normalizer-bandgap.pkl                      # ///
│   ├── best_ddec                                       # ///
│   │   ├── ddec.pth                                    # ///
│   │   └── normalizer-ddec.pkl                         # ///
│   ├── best_ddec_COF                                   # ///
│   │   ├── ddec.pth                                    # ///
│   │   └── normalizer-ddec.pkl                         # ///
│   ├── best_pbe                                        # ///
│   │   ├── pbe-atom.pth                                # ///
│   │   └── normalizer-pbe.pkl                          # ///
    ├── chk_bader                                       # Bader
│   │   └── checkpoint.pth                              # Checkpoint of bader
│   ├── chk_bandgap                                     # Bandgap
│   │   └── checkpoint.pth                              # Checkpoint of bandgap
    ├── chk_cm5                                         # CM5
│   │   └── checkpoint.pth                              # ///
│   ├── chk_ddec                                        # ///
│   │   └── checkpoint.pth                              # ///
│   └── chk_pbe                                         # ///
│       └── checkpoint.pth                              # ///
│
├── PACMaN.py                                           # main python file for atomic charge assignment by command line
├── LICENSE.txt                                         # MIT license
├── README.md                                           # Usage/Source
├── predict_E.py                                        # main python file for energy/bandgap prediction
├── predict_ddec.py                                     # main python file for atomic charge prediction
├── requirements.txt                                    # packages need to be installed
├── train_E.py                                          # main python file for energy/bandgap training
└── train_ddec.py                                       # main python file for atomic charge training

```
 
**Group:**   [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home?authuser=0)                                
