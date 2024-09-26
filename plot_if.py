import model
import json
import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from model import *

with open('params1606.json', 'r') as file:
    PARAMS = json.load(file)
print(PARAMS)

params = dict(PARAMS)
params['chi'] = 0
params['LENNART'] = 1
params['Iint'] = 92e-9 + 73e-9

@functools.partial(jax.vmap, in_axes=[0, None])
def fI_curve(Iinj, params):
    nsteps = 50000
    dt = 0.02*ms
    calibration=200*ms
    Iexc = jnp.zeros(nsteps)
    IinjnA = Iinj * nA
    Iexc = Iexc.at[int(calibration/dt):].set(IinjnA)
    stims = Stim(
        Ipf1=jnp.zeros(nsteps),
        Ipf2=jnp.zeros(nsteps),
        Ipf3=jnp.zeros(nsteps),
        Icf=jnp.zeros(nsteps),
        Iexc=Iexc,
        Iinh=jnp.zeros(nsteps)
    )
    params = Params.init(params)
    state0 = State.init(params)
    trace = model.simulate(state0, stims, params, dt)
    return trace.sspike[int(calibration/dt):].sum() / (nsteps*dt - calibration), trace

f = plt.figure()

Is = jnp.linspace(-10,350,30)
F, trace = fI_curve(Is, params)

plt.plot(Is, F, color='black')
plt.gca().spines['top left bottom right'.split()].set_visible(False)
plt.xlabel('Stimulation current (nA)')
plt.ylabel('Frequency (Hz)')
#plt.xticks([50, 100, 150, 200, 250, 300])
plt.xticks([0, 100, 200, 300])
plt.yticks([0, 100, 200, 300])
plt.show()
