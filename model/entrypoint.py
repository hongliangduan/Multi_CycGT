#!/usr/bin/env python
import os

if __name__ == '__main__':
    cwd = os.getcwd()
    for file in os.listdir(cwd):
        pathname = os.path.join(cwd, file)
        if not os.path.isdir(pathname):
            continue
        else:
            # folder
            os.system(f"cd {pathname}; env python {os.path.join(pathname, 'model_concat.py')}")
            pass

