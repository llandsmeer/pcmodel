import model
import json
import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from model import *

RANDOMIZE_PARAMS = (
    'a', 'b', 'vreset', 'wreset', 'eps', 'Iint', 'C', 'v_sp', 'd_z', 'el',
    'vth', 'g_sd', 'g_ds', 'ad', 'DeltaV', 'vthd', 'vths', 'vd_sp', 'Cd',
    'wdreset', 'wiggle_vd', 'DeltaCaV', 'max_alpha', 'Kcf', 'max_vCas',
    'Kpf', 'max_vCad', 'Ca_half', 'tauw', 'tauCa', 'tauR', 'zmax', 'c',
    'l', 'm', 'g1', 'g2', 'g3', 'eps_z0', 'Deltaeps_z', 'dres_coeff',
    'dsp_coeff', 'dsp_coeff2', 'slp', 'n', 'gl', 'asd', 'vsd3reset',
    'vsd2reset', 'wsdreset', 'vsd_sp', 'Csd', 'g0', 'DeltaVsd', 'vth3',
    'vth2', 'sdsf3', 'sdsf2', 'sdsf0', 'gamma1', 'gamma2', 'gamma3',
    'eta_z', 'p', 'w00', 'tau_s2', 'tau_s3')

with open('params1606.json', 'r') as file:
    PARAMS = json.load(file)
print(PARAMS)

params = dict(PARAMS)
params['chi'] = 0
params['LENNART'] = 0
params['Iint'] = 92e-9 + (73e-9 if params['LENNART'] else 0)

#@functools.partial(jax.vmap, in_axes=[0, None])
def cf_trace(params):
    nsteps = 50000
    dt = 0.02*ms
    tcf = 500
    t = dt * jnp.arange(nsteps)
    calibration=200*ms
    stims = Stim(
        Ipf1=jnp.zeros(nsteps),
        Ipf2=jnp.zeros(nsteps),
        Ipf3=jnp.zeros(nsteps),
        Icf=(10 * nA * ((t % 1 > tcf * ms) & (t % 1 < (tcf+5) * ms))),
        Iexc=jnp.zeros(nsteps),
        Iinh=jnp.zeros(nsteps)
    )
    params = Params.init(params)
    state0 = State.init(params)
    trace = model.simulate(state0, stims, params, dt)
    return trace.sspike[int(calibration/dt):], trace

f = plt.figure()

Is = jnp.linspace(-10,350,30)

nsamples = 100
rand = jax.random.uniform(
        key=jax.random.PRNGKey(0),
        shape=(len(params), nsamples),
        minval=0.95,
        maxval=1.05)
ones = jnp.ones(nsamples)
sampled_params = {
        k: v * (rand[i] if k in RANDOMIZE_PARAMS else ones)
        for i, (k, v) in enumerate(params.items())
        }

spikes, traces = jax.vmap(cf_trace)(sampled_params)
# spikes : (nsamples, ntimesteps)

# 25600

spike_times = [
    jnp.where(spikes[i])[0]
    for i in range(nsamples)
    ]
spike_times = sorted(
    spike_times,
    key = lambda idx: idx[idx>16500].min().item()
    )

for i, idx in enumerate(spike_times):
    st = idx * 0.02
    plt.plot(
            st,
            jnp.full(st.shape, i),
            'o', color='black')



#for i in
#    plt.plot(traces.state.Vs[i]+i/10, color='black')

plt.show()

# plt.plot(Is, F, color='black')
# plt.gca().spines['top left bottom right'.split()].set_visible(False)
# plt.xlabel('Stimulation current (nA)')
# plt.ylabel('Frequency (Hz)')
# #plt.xticks([50, 100, 150, 200, 250, 300])
# plt.xticks([0, 100, 200, 300])
# plt.yticks([0, 100, 200, 300])
# plt.show()
