import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def f1(h):
    return 20*(1+(0.15*((h/4400)**4)))

def f2(h):
    return 10*(1+(0.15*(((7000-h)/2200)**4)))

def f3(h):
    return 50*(1+(0.15*((h/4400)**4)))

h = np.linspace(0, 7500, 7500)
f1h = [f1(hh) for hh in h]
f2h = [f2(hh) for hh in h]
f3h = [f3(hh) for hh in h]

fig, ax = plt.subplots(figsize=(16,9), dpi=150)
plt.rc('text', usetex=True)
ax.plot(h, f1h, label=r'\[f_1(h) = 20\left(1+0.15\left(\frac{h}{4400}\right)^4\right)\]')
ax.plot(h, f2h, label=r'\[f_2(h) = 10\left(1+0.15\left(\frac{7000-h}{2200}\right)^4\right)\]')
ax.plot(h, f3h, label=r'\[f_3(h) = 50\left(1+0.15\left(\frac{h}{4400}\right)^4\right)\]')
plt.legend(loc=0)
plt.xlabel('Path flow (h1)')
plt.ylabel('Time')
plt.title('Time vs Path Flow')
plt.show()
