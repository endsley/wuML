#!/usr/bin/env python

import os
import time

os.system('rm -r dist/')
os.system('python setup.py sdist')
os.system('twine upload dist/*')
#time.sleep(2)
#os.system('pip install wuml -U')

