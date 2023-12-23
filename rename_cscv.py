import os
import re

if __name__ == '__main__':

    path = '/home/tanshui/work/Multi_CycGT/pred_data/machine_learing/svm_tanimoto/svm_tanimoto_pred_data'
    mlist = os.listdir(path)
    # print(mlist)

    # mlist.sort(key=lambda x: int(x[:-12]))
    m = re.compile(f'\\d+')
    for name in mlist:
        # num = i[1]
        num = m.search(name).group()

        os.system(f'mv {os.path.join(path, name)} {os.path.join(path, f"experiment_{num}_predicted_values.csv")}')
