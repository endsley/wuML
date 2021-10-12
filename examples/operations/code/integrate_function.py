#!/usr/bin/env python

import wuml 

def f(x):
	if x < 0: return 0
	if x > 1: return 0

	return 1

[area, error] = wuml.integrate(f, 0, 0.5)
print('Area: %.4f, error: %.4f'%(area, error))

