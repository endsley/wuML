
from scipy.integrate import quad

def integrate(foo, x0, x1):
	[result, error] = quad(foo, x0, x1)
	return [result, error]

