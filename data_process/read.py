



import pandas as pd
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')



from rdkit import Chem

import numpy as np
def read_mol_file(filename):
    """
    读取mol文件中原子的坐标
    :param filename: mol文件名
    :return: 原子坐标列表
    """
    mol = Chem.MolFromMolFile(filename, sanitize=False)
    atoms = mol.GetAtoms()
    atom_coords = []
    for atom in atoms:
        atom_coord = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_coords.append([atom_coord.x, atom_coord.y, atom_coord.z])
    return atom_coords
idarr = np.zeros((data.shape[0],3))
mlist = []
mdict = {}
max= 1


def read_pos():
    pos_dic = {}
    m_list =[]
    for i in range(data['CycPeptMPDB_ID'].shape[0]):
        id_num = data['CycPeptMPDB_ID'][i]
        iddata = np.array(read_mol_file('../cycy/id_{}.mol'.format(id_num)))

        iddata_pad = iddata.reshape(1,iddata.shape[0] * iddata.shape[1])

        if iddata_pad.shape[1] < 381:
            iddata_pad = np.concatenate([iddata_pad,np.array([[0]*(381-iddata_pad.shape[1])])],axis=1)
        pos_dic[id_num] = iddata_pad
        m_list.append(iddata_pad)

    return pos_dic,np.array(m_list)
