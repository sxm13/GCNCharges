from ase.io import read,write
import pymatgen.core as mg

mof = "./test_cubtc/Cu-BTC_gcn.cif"
struc = mg.Structure.from_file(mof)
print(struc)
struc = read(mof)
print(struc)