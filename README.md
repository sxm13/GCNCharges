<h1 align="center">GCN Charge</h1>

<h4 align="center">

</h4>              

A MOF/COF charge predicter by **G**raph **C**onvolution **N**etwork.                           

[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads) [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.123456-blue)](https://doi.org/10.5281/zenodo.123456)  [![MIT](https://img.shields.io/badge/License-MIT-blue.svg)]() [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sxmzhaogb@gmail.com) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)]() [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)]()

![Logo](/figs/toc.jpg)              

# Download

```sh
git clone https://github.com/sxm13/GCNCharges.git
```   

# Installation

```sh
pip install -r requirements.txt
```

# Charge Assignment               
You can put your cif files in any folder, but please run the code and jupyter notebook in this folder.                

**bash**
```sh
python GCNCharge.py [folder name] [MOF/COF]
```
example: ```python GCNCharge.py test_file MOF```

**notebook**
```sh
import GCNCharge4notebook
GCNCharge4notebook.GCNChagre(file="./test/test_cubtc/",model="MOF")
```
file: your folder contains cif files                               
model: MOF or COF                                                   
there is an example in ```GCNCharge.ipynb```

# Website & Zenodo
IF you do not want to install GCN Charge, you can go to this :point_right: [link](https://gcn-charge-predicter-mtap.streamlit.app/)       
IF you want to DOWNLOAD code and full dataset, you can go to this :point_right: [link](https://zenodo.org/records/)             

# Reference
If you use CGCN Charge, please cite [this paper]():
```bib
@article{,
    title={},
    DOI={},
    journal={},
    author={},
    year={},
    pages={}
}
```

## Development & Bugs

 If you encounter any problem during using ***GCN Charge***, please talk to me ```sxmzhaogb@gmail.com```.                            

 
**Group:**   [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home?authuser=0)                                
