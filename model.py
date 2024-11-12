import jax
import typing
import functools
import jax.numpy as jnp
from jax.numpy import exp

__all__ = [ 'uS', 'mV', 'nA', 'nF', 'ms', 'Stim', 'Params', 'State']

uS = 1e-6; mV = 1e-3; nA = 1e-9; nF = 1e-9; ms = 1e-3

# Input currents into Purkinje Cells
class Stim(typing.NamedTuple):
    Ipf1: float | jax.Array
    Ipf2: float | jax.Array
    Ipf3: float | jax.Array
    Icf : float | jax.Array
    Iexc: float | jax.Array
    Iinh: float | jax.Array

# Model's Parameters definition. Given a default value but can also be assigned
# by importing a dictionary.
class Params(typing.NamedTuple):
    a: float
    b: float
    vreset: float
    wreset: float
    eps: float
    Iint: float
    C: float
    v_sp: float
    d_z: float
    el: float
    vth: float
    g_sd: float
    g_ds: float
    ad: float
    DeltaV: float
    vthd: float
    vths: float
    vd_sp: float
    Cd: float
    wdreset: float
    wiggle_vd: float
    DeltaCaV: float
    max_alpha: float
    Kcf: float
    max_vCas: float
    Kpf: float
    max_vCad: float
    Ca_half: float
    tauw: float
    tauCa: float
    tauR : float
    zmax: float
    chi : float
    c: float
    l: float
    m: float
    g1: float
    g2: float
    g3: float
    eps_z0: float
    Deltaeps_z: float
    dres_coeff: float
    dsp_coeff : float
    dsp_coeff2: float
    slp: float
    n: float
    gl: float
    asd: float
    vsd3reset: float
    vsd2reset: float
    wsdreset: float
    vsd_sp: float
    Csd: float
    g0: float
    DeltaVsd: float
    vth3: float
    vth2: float
    sdsf3: float
    sdsf2: float
    sdsf0: float
    gamma1: float
    gamma2: float
    gamma3: float
    eta_z: float
    p: float
    w00: float
    tau_s2: float
    tau_s3: float
    LENNART: float
    @classmethod
    #define the import of a dictionary to give the parameters their (scaled) values
    def init(cls, d: typing.Dict[str, float]):
        return cls(**d)
    @classmethod
    #Define default values of the parameters
    def makedefault(cls):
        default = {
            'a': 0.1*uS, 'b': -3*uS, 'vreset': -55*mV, 'wreset': 15*nA, 'eps': 1, 'Iint': 91*nA, 'C': 2*nF,
            'v_sp': -5*mV, 'd_z': 40*nA, 'el': -61*mV, 'vth':  -45*mV, 'g_sd': 3*uS, 'g_ds': 6*uS,
            'ad': 2*uS, 'DeltaV':  1*mV, 'vthd':  -40*mV, 'vths':  -42.5*mV, 'vd_sp': -35*mV, 'Cd': 2*nF, 'wdreset': 15*nA, 'wiggle_vd': 3*mV,
            'DeltaCaV': 5.16*mV, 'max_alpha':  0.09, 'Kcf': 1*nA, 'max_vCas': 80*(nA/ms), 'Kpf': 0.5*nA, 'max_vCad': 1.5*(nA/ms), 'Ca_half': 50*nA,
            'tauw': 1*ms, 'tauCa': 100*ms, 'tauR': 75*ms, 'zmax': 100*nA, 'chi':0,
            'c': 0.1*(1/mV), 'l': 0.1*(1/nA), 'm': 0.2*(1/nA), 'g1': 4*uS, 'g2': 4*uS, 'g3': 4*uS,
            'eps_z0': 10, 'Deltaeps_z': 8, 'dres_coeff': 2,'dsp_coeff':0.5, 'dsp_coeff2':0.1, 'slp': 5e-4*(1/nA), 'n': 0.2*(1/nA),
            'gl': 0.1*uS, 'asd': 0.1*uS, 'vsd3reset': -55*mV, 'vsd2reset': -50*mV, 'wsdreset': 15*nA, 'vsd_sp': -20*mV, 'Csd': 5*nF,
            'g0': 1*uS, 'DeltaVsd': 5*mV, 'vth3': -42.5*mV, 'vth2': -42.5*mV, 'sdsf3': 3.15*uS, 'sdsf2': 2.37*uS, 'sdsf0': 5,
            'gamma1': 25, 'gamma2': 25, 'gamma3': 25, 'eta_z':0.75, 'p': 2, 'w00': 4*nA, 'tau_s2': 150*ms, 'tau_s3': 75*ms,
            'LENNART': 0.
        }
        return cls.init(default)

#Definition of state variables to track as a function of time
class State(typing.NamedTuple):
    Vs:   float | jax.Array
    Vd:   float | jax.Array
    vd1:  float | jax.Array
    vd2:  float | jax.Array
    vd3:  float | jax.Array
    sf2:  float | jax.Array
    sf3:  float | jax.Array
    w0:   float | jax.Array
    z:    float | jax.Array
    dres: float | jax.Array
    Cas:  float | jax.Array
    Cad:  float | jax.Array
    ws:   float | jax.Array
    wd:   float | jax.Array
    wd2:  float | jax.Array
    wd3:  float | jax.Array
    wd1:  float | jax.Array
    eps_z:float | jax.Array
    act:  float | jax.Array
    t:    float | jax.Array
    t_ds2:float | jax.Array
    t_ds3:float | jax.Array
    sf3:  float | jax.Array
    alpha:float | jax.Array

    @classmethod
    #Give initial values to state variables
    def init(cls, params: Params):
        return cls(
            Vs=params.el, Vd=params.el, vd1=params.el, vd2=params.el, vd3=params.el,
            sf2=params.sdsf2, sf3=params.sdsf3, w0=params.w00, z=0, eps_z=params.eps_z0,
            dres=0, Cas=0, Cad=0, ws=0, wd=0, wd2=0, wd3=0, wd1=0, act=0, t=0.0, t_ds2=0.0,
            t_ds3=0.0, alpha=0)

# Accounting of Spikes in electrical activity: Dendritic Calcium spikes at dendritic compartments
# and Action Potentials at the Soma
class Trace(typing.NamedTuple):
    state: State
    sspike: bool  | jax.Array | int
    dspike: bool  | jax.Array | int
    d3spike: bool | jax.Array | int
    d2spike: bool | jax.Array | int

@functools.partial(jax.jit, static_argnames=['dt'])
def timestep(params: Params, state: State, stim: Stim, dt: float):
    t       = state.t+dt
    I       = params.Iint + stim.Iinh + stim.Iexc
    Ipf     = params.gamma1*stim.Ipf1+params.gamma2*stim.Ipf2+params.gamma3*stim.Ipf3
    zeff    = state.z + state.dres
    zlim    = params.eta_z*(params.zmax+(jnp.where(I > params.Iint, 1, 0)*params.chi*(I-params.Iint)))
    #zlim    = params.eta_z * params.zmax * ( 1 + (jnp.where(I > params.Iint, 1, 0) * params.chi * ((I-params.Iint)/nA)**2*2e-5))
    eps_z_old   = params.eps_z0 - params.Deltaeps_z/(1+exp(-params.l*(zeff-zlim)))
    eps_z_new   = params.eps_z0 - params.Deltaeps_z * 1.0
    eps_z = eps_z_old * (1-params.LENNART) + eps_z_new * params.LENNART
    #Ca-related variables
    Ca      = state.Cas + state.Cad
    act     = 1/(1+exp(-params.m*(Ca-params.Ca_half)))
    vCas    = params.max_vCas*stim.Icf / (params.Kcf+stim.Icf)
    vCad    = params.max_vCad*Ipf / (params.Kpf+Ipf)
    alpha   = (Ca<params.Ca_half)* ( params.slp*(params.Ca_half-Ca) + params.max_alpha / (1+exp(-params.n*(Ca-params.Ca_half))) ) + (Ca>params.Ca_half)* ( params.max_alpha / (1+exp(-params.n*(Ca-params.Ca_half))) )
    # derivatives
    dot_Vs  = (((params.el-state.Vs)**2*uS**2 + params.b*(state.Vs-params.el)*state.ws  - state.ws**2)/nA + I - zeff + params.g_sd*(state.Vd-state.Vs) )/params.C
    dot_ws  = params.eps*(params.a*(state.Vs-params.el)-state.ws+state.w0-alpha*Ca)/params.tauw
    dot_Vd  = (params.g_ds*(state.Vs-state.Vd+params.wiggle_vd)+params.g1*(state.vd3-state.Vd)+params.g1*(state.vd2-state.Vd)+params.g1*(state.vd1-state.Vd) + params.sdsf0*(params.DeltaV)*exp((state.Vd-params.vth)/(params.DeltaV))*uS-state.wd)/params.Cd
    dot_wd  = (params.ad*(state.Vd-params.el)-state.wd)/params.tauw
    dot_z   = -eps_z*state.z/params.tauCa
    dot_dres= act**params.p*params.dres_coeff*nA/ms-state.dres/params.tauR
    dot_Cas = vCas / (1+exp(-params.c*(state.Vs-params.vths))) - state.Cas/params.tauCa
    dot_Cad = vCad*(exp((state.vd3-params.vthd)/(params.DeltaCaV)) * (stim.Ipf3!=0)  + exp((state.vd2-params.vthd)/(params.DeltaCaV)) * (stim.Ipf2!=0) + exp((state.vd1-params.vthd)/(params.DeltaCaV)) * (stim.Ipf1!=0)) - state.Cad/params.tauCa
    dot_vd3 = (params.g0*(state.Vd-state.vd3)+params.gl*(params.el-state.vd3+params.wiggle_vd)+state.sf3*(params.DeltaVsd)*exp((state.vd3-params.vth3)/(params.DeltaVsd))-state.wd3+params.gamma3*stim.Ipf3)/params.Csd
    dot_wd3 = (params.asd*(state.vd3-params.el)-state.wd3)/(params.tauw)
    dot_vd2 = (params.g0*(state.Vd-state.vd2)+params.gl*(params.el-state.vd2+params.wiggle_vd)+state.sf2*(params.DeltaVsd)*exp((state.vd2-params.vth2)/(params.DeltaVsd))-state.wd2+params.gamma2*stim.Ipf2)/params.Csd
    dot_wd2 = (params.asd*(state.vd2-params.el)-state.wd2)/params.tauw
    dot_vd1 = (params.g0*(state.Vd-state.vd1)+params.gl*(params.el-state.vd1+params.wiggle_vd)-state.wd1+params.gamma1*stim.Ipf1)/params.Csd
    dot_wd1 = (params.asd*(state.vd1-params.el)-state.wd1)/params.tauw
    # events
    sspike = state.Vs > params.v_sp
    dspike = state.Vd > params.vd_sp
    d3spike = state.vd3 > params.vsd_sp
    d2spike = state.vd2 > params.vsd_sp
    # updates
    Vs = jax.lax.select(sspike, params.vreset, state.Vs + dot_Vs*dt)
    ws = jax.lax.select(sspike, params.wreset, state.ws + dot_ws*dt)
    Vd = jax.lax.select(dspike, params.vreset, state.Vd + dot_Vd*dt)
    wd = state.wd + dot_wd*dt
    z = state.z + dot_z*dt
    dres = state.dres + dot_dres*dt
    Cas = state.Cas + dot_Cas*dt
    Cad = state.Cad + dot_Cad*dt
    vd3 = jax.lax.select(d3spike, params.vsd3reset, state.vd3 + dot_vd3*dt)
    wd3 = state.wd3 + dot_wd3*dt
    vd2 = jax.lax.select(d2spike, params.vsd2reset, state.vd2 + dot_vd2*dt)
    wd2 = state.wd2 + dot_wd2*dt
    vd1 = state.vd1 + dot_vd1*dt
    wd1 = state.wd1 + dot_wd1*dt
    z = z \
          + sspike * (params.d_z) \
          + d3spike * (params.dsp_coeff*params.d_z) \
          + d2spike * (params.dsp_coeff*params.d_z)
    dres= dres \
            + d3spike * (params.dsp_coeff2*params.d_z) \
            + d2spike * (params.dsp_coeff2*params.d_z)
    wd = jax.lax.select(dspike, wd + params.wdreset, wd)
    wd3 = jax.lax.select(d3spike, wd3 + params.wsdreset, wd3)
    wd2 = jax.lax.select(d2spike, wd2 + params.wsdreset, wd2)
    # ignored?
    t_ds3=state.t_ds3
    t_ds3=jax.lax.select(d3spike, state.t, state.t_ds3)
    t_ds2=state.t_ds2
    t_ds2=jax.lax.select(d2spike, state.t, state.t_ds2)
    sf2 = params.sdsf2*(1-0.9*exp(-(state.t-state.t_ds2)/params.tau_s2))
    sf3 = params.sdsf3*(1-0.9*exp(-(state.t-state.t_ds3)/params.tau_s3))
    w0 = state.w0
    state_next = State(Vs=Vs, Vd=Vd, vd1=vd1, vd2=vd2, vd3=vd3, sf2=sf2, sf3=sf3, w0=w0, z=z, dres=dres, alpha=alpha,
                       Cas=Cas, Cad=Cad, ws=ws, wd=wd, wd2=wd2, wd3=wd3, wd1=wd1, eps_z=eps_z, act=act, t=t, t_ds3=t_ds3, t_ds2=t_ds2)
    trace = Trace(state=state, sspike=sspike, dspike=dspike, d3spike=d3spike, d2spike=d2spike)
    return state_next, trace

@jax.jit
def simulate(state0, stims, params, dt):
    _, trace = jax.lax.scan(
        lambda state, stim: timestep(params, state, stim, dt),
        state0, stims)
    return trace
