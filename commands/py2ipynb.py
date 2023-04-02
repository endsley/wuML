#!/usr/bin/env python

import os
import sys

pth = sys.argv[1]

f = open(pth, 'r')
code = f.read()
newCode = code.replace('\n\n', '\n#\n')

newPy = 'k' + pth
f = open(newPy, 'w')
f.write(newCode)
f.close()


os.system('p2j -o ' + newPy)
kipynb = newPy.replace('.py', '.ipynb')
ipynb = pth.replace('.py', '.ipynb')

os.system('jupyter nbconvert --to notebook --execute ' + kipynb)
ran_ipynb = kipynb.replace('.ipynb', '.nbconvert.ipynb')

os.system('rm ' + kipynb)
os.system('rm ' + newPy)
os.system('mv ' + ran_ipynb + ' ' + ipynb)
