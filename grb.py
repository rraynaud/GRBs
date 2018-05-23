#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Transient Lightcurve Modelling

References:
-----------
    - Zhang and Meszaros, 2001, ApJ, 552
    - Gompertz et al., 2014, MNRAS, 438
    - Sun et al., 2017, ApJ, 835:7
    - Ai et al., arXiv:1802.00571v1

"""
__author__="GRBs: Guilet, Raynaud, Bugli"
__email__ ="jerome.guilet@cea.fr ; raphael.raynaud@cea.fr ; matteo.bugli@cea.fr"
####################################
import os,sys
import numpy as np
import matplotlib as mpl
import itertools
from scipy.integrate import odeint
#from astropy.io import fits
from astropy.table import Table
import warnings
try:
    import magic
    mpl.rcParams.update(mpl.rcParamsDefault)
except ImportError:
    pass
###########################
### plot parameters
###########################
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
plt.rcParams["axes.formatter.limits"] = [-2,2]
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams["axes.titlesize"] = 'xx-large'
plt.rcParams["xtick.labelsize"] = 'x-large'
plt.rcParams["ytick.labelsize"] = 'x-large'
###########################################
### dictionary units
### each key must be an input parameter
###########################################
d_units = {}
d_units['t_min'] = 's'
d_units['t_max'] = 's'
d_units['t_num'] = ''
d_units['NS_B'] = 'G'
d_units['NS_mass'] = 'g'
d_units['NS_radius'] = 'cm'
d_units['NS_period'] = 's'
d_units['NS_eta_dip'] = ''
d_units['NS_critical_beta'] = ''
d_units['AG_T0'] = 's'
d_units['AG_Eimp'] = 'erg'
d_units['AG_alpha'] = ''
d_units['DISK_mass0'] = 'g'
d_units['DISK_radius'] = 'cm'
d_units['DISK_alpha'] = ''
d_units['DISK_aspect_ratio'] = ''
d_units['DISK_eta_prop'] = ''
d_units['EOS_Mtov'] = 'g'
d_units['EOS_alpha'] = ''
d_units['EOS_beta'] = ''
d_units['EOS_I'] = 'g cm^2'
d_units['EJECTA_mass'] = 'g'
d_units['EJECTA_opacity'] = 'cm^2/g'
d_units['EJECTA_heating_efficiency']=''
d_units['EJECTA_Gamma0'] = ''
d_units['EJECTA_co_T0'] = 's'
d_units['EJECTA_co_TSIGMA'] = 's'
d_units['EJECTA_co_Time0'] = 's'
d_units['EJECTA_co_Eint0'] = 'erg'
d_units['EJECTA_co_Volume0'] = 'cm^3'
d_units['EJECTA_radius0']= 'cm'
d_units['EJECTA_theta']= 'rad'
d_units['tag']=''
###########################################
## EOS database (Ai 2018, Table 1)
## to be completed
###########################################
EOS = {}
EOS['GM1']   = {'EOS_Mtov' :2.37,
                'EOS_alpha':1.58e-10, 
                'EOS_beta' :-2.84,
                'EOS_I'    :3.33e45,
                'NS_radius':12.05e5,}
EOS['Shen']  = {'EOS_Mtov' :2.18,
                'EOS_alpha':4.678e-10, 
                'EOS_beta' :-2.738,
                'EOS_I'    :4.675e45,
                'NS_radius':12.40e5,}
EOS['BSk21'] = {'EOS_Mtov' :2.28,
                'EOS_alpha':2.81e-10, 
                'EOS_beta' :-2.75,
                'EOS_I'    :4.37e45,
                'NS_radius':11.08e5,}
EOS['DD2']   = {'EOS_Mtov' :2.42,
                'EOS_alpha':1.37e-10, 
                'EOS_beta' :-2.88,
                'EOS_I'    :5.43e45,
                'NS_radius':11.89e5,}
EOS['DDME2'] = {'EOS_Mtov' :2.48,
                'EOS_alpha':1.966e-10, 
                'EOS_beta' :-2.84,
                'EOS_I'    :5.85e45,
                'NS_radius':12.09e5,}
EOS['CDDM1'] = {'EOS_Mtov' :2.21,
                'EOS_alpha':3.93e-16,
                'EOS_beta' :-5.0,
                'EOS_I'    :11.67e45,
                'NS_radius':13.99e5}
###########################################
def Generate_inputs(dico):
    """
    Input: dictionary of the form
        {key: array of values to explore}

    Return:
        a list of dictionaries

    """
    keys = dico.keys()
    comb = tuple(itertools.product(*dico.values()))
    num = len(comb)

    out = [{akey: aval for akey,aval in zip(keys,vals)}
           for vals in comb]

    ### add a tag to differentiate the models
    tags = ['m'+str(i+1).rjust(len(str(num)),'0') for i in range(num)]
    for i,adico in enumerate(out):
        adico['tag'] = tags[i]

    print ('Generating %i models'%num)
    return out


###########################################
class GRB(object):
    """This class defines a transient lightcurve model. 
    It implements a modified version of Sun et al. 2017, 
    with Xray emission from a free zone and a trapped zone.
    The spindown luminosity of the NS takes into account 
    contributions from standard dipolar spindown and the 
    propeller model.
    
    The main outputs are stored in the following class members:
    LX_free --> Free zone luminosity
    LX_trap --> Trapped zone luminosity
    L_dip   --> Dipolar spindown luminosity
    L_prop  --> Propeller spindown luminosity

    Other time-dependent quantities are available, such as
    characteristic radii, torques, optical depth and temperature 
    of the ejecta, etc.

    Notes
    -----
    The code works in CGS_Gaussian units
    Methods of the form Eval_* set attributes
    Methods of the form L_* return a timeserie (1D array)

    Ejecta parameters for the trapped zone:
        *_co_* = quantity defined in the co-moving frame

    Time dependent variables:
        - Omega (neutron star angular velocity)
        - Gamma (Lorentz factor)
        - Radius (radius of the ejecta)
        - co_Time : co-moving time
        - co_Eint (ejecta internal energy)
        - c_Volume (volume of the ejecta)

    """
    def __init__(self,
                 t_min=0,
                 t_max=10,
                 t_num=200,
                 NS_B=1e15,
                 NS_mass=1.4,
                 NS_radius=12.4e5, # Shen EOS as default
                 NS_period=np.inf, # automatic determination
                 NS_eta_dip=0.05,
                 NS_critical_beta=0.27, # bar-mode instability criterion
                 AG_T0=10,
                 AG_Eimp=-np.inf,
                 AG_alpha=0,
                 DISK_mass0=2.01e-2,
                 DISK_radius=5.e8, # 5000 km
                 DISK_alpha=0.1,   # disk viscosity parameter
                 DISK_aspect_ratio=0.3, # Aspect ratio H/R
                 DISK_eta_prop=0.4,
                 EOS_Mtov=2.18, # Msun
                 EOS_alpha=4.678e-10,
                 EOS_beta=-2.738,
                 EOS_I=4.37e45,
                 EJECTA_mass=0.01,
                 EJECTA_opacity=2,
                 EJECTA_heating_efficiency=0.5,
                 EJECTA_theta=0.,
                 EJECTA_Gamma0=1.,
                 EJECTA_co_T0=1.3, # eq. 15 Sun (2017)
                 EJECTA_co_TSIGMA=0.11,
                 EJECTA_co_Time0=1.,
                 EJECTA_co_Eint0=1e48,
                 EJECTA_co_Volume0=4./3.*np.pi*1e24,
                 EJECTA_radius0=1e10, #10^5 km
                 tag='notag',
                 verbose=True):
        """
        Parameters (in cgs units, when not specified otherwise)
        ----------

        t_min, t_max, t_num : float, float, int
                define the integration time

        NS_B : float
                magnetar magnetic field
        
        NS_period : float
                magnetar period

        NS_mass : float
                magnetar mass (in units of solar mass)

        NS_radius : float
                magnetar radius

        NS_eta_dip : float
                dipole efficiency factor                 
        
        DISK_mass : float
                disk mass (in units of solar mass)

        DISK_radius : float
                disk radius

        DISK_alpha : float
                disk viscosity parameter

        DISK_aspect_ratio : float
                disk aspect ratio H/R

        DISK_eta_prop : float
                propeller efficiency factor

        EOS_Mtov : float
                maximum mass of a NS with zero spin

        EOS_alpha : float
                phenomenological parameter used
                to compute the NS maximum mass

        EOS_beta : float
                similar to EOS_alpha

        tag : string

        verbose : boolean
                print a summary when instantiating 
                a GRB object
 
        """
        super(GRB, self).__init__()
        ############################
        ### save control parameters
        ############################
        self.parameters = locals()
        ### remove useless parameters
        del self.parameters['verbose']
        del self.parameters['self']
        if sys.version_info.major==3:
            del self.parameters['__class__']

        #################################
        ### astrophysical constants (CGS)
        #################################
        self.lightspeed = 299792458e2 # cm/s
        self.Msun = 1.98855e33 # g
        ### gravitational constant
        self.gravconst = 6.67259e-8 # cm^3 g^-1 s^-2
        self.hPlanck = 6.6260755e-27
        self.kBoltzmann = 1.380658e-16
        self.radiation_const = 7.5646e-15
        self.Ev_to_Hz = 1.602176565e-12/self.hPlanck

        ##########################
        ## define integration time
        ##########################
        self.time = np.logspace(t_min,t_max,t_num)
        self.time_units = 's'

        ##############################
        ## automatic attribute setting
        ##############################
        for key,val in self.parameters.items():
            setattr(self,key,val)
            key2 = key+'_units'
            setattr(self,key2,d_units[key])
        #####################
        ## rescaling masses !
        #####################
        self.NS_mass     *= self.Msun
        self.DISK_mass0  *= self.Msun
        self.EOS_Mtov    *= self.Msun
        self.EJECTA_mass *= self.Msun

        ######################
        ## derived quantities
        ######################
        ## uncomment to use
        ## Gompertz's definition
        #self.Eval_MomentOfInertia()
        ######################
        self.Eval_Omega0(verbose)
        self.Eval_T_em()
        #self.Eval_L_em0()
        #self.Eval_Tc()
        self.Eval_magnetic_moment()
        self.Eval_OmegaKep()
        self.Eval_viscous_time()
        self.Eval_Mdot0()
        self.Eval_critical_angular_velocity()
        ######################
        ## fine tuning
        ######################
        #print ('Fine tuning ON...')
        #self.AG_Eimp = self.L_em0#*self.T0

        ######################
        ## Time integration
        ######################
        self.Time_integration(self.time)

        ######################
        ### Light curves
        ######################
        ## outputs
        self.Eval_LX_free(self.time)
        self.Eval_LX_trap()

        ######################
        ### Further outputs
        ######################
        self.Eval_radii(self.time)
        self.Eval_torques(self.time)
        self.Eval_diagnostic_outputs(self.time)
        self.Eval_T_tau(self.time) # ejecta become optically thin
        self.Eval_T_col(self.time) # NS collapse

        ######################
        ### print a summary
        ######################
        if verbose is True:
            self.Info()

    ##########################################################
    ### DEFINITION OF METHODS
    ##########################################################
    def Info(self):
        """print a summary"""
        control_param = list(self.parameters.keys())
        control_param.remove('tag')
        control_param.remove('t_min')
        control_param.remove('t_max')
        control_param.remove('t_num')

        derived_param = ['time_collapse','time_opacity',
                         'critical_period','OmegaKep',
                         'time_spindown','viscous_time']

        ### for the layout column width
        lenun = max([len(getattr(self,afield+'_units'))
                     for afield in control_param+derived_param])
        lensy = max([len(afield)
                     for afield in control_param+derived_param])
        
        header = '{:-^%i}'%(lenun+lensy+2+8+1)
        ligne = '{:%i} {: 8.2e} {:%i}'%(lensy,lenun)
        
        print (header.format('Model properties'))
        print (header.format('Input parameters'))
        for afield in sorted(control_param):
            info = ligne.format(afield,
                                getattr(self,afield),
                                getattr(self,afield+'_units'))
            print (info)

        print (header.format('Derived quantities'))
        for afield in sorted(derived_param):
            info = ligne.format(afield,
                                getattr(self,afield),
                                getattr(self,afield+'_units'))
            print (info)
        print(header.format('-'))
    #################################################
    ### Evaluation of the derived constant parameters
    #################################################
    def Eval_MomentOfInertia(self):
        """
        Set the magnetar moment of inertia
        """
        #################################
        ### normalisation
        #################################
        ### full sphere formula
        #norm = 2./5
        ### Gompertz (2014)
        norm = 0.35
        #################################
        self.EOS_I = norm * self.NS_mass*self.NS_radius**2

    def Eval_magnetic_moment(self):
        """
        compute the magnetar magnetic moment
        """
        self.mu = self.NS_B * self.NS_radius**3
        self.mu_units = "G cm^3"

    def Eval_OmegaKep(self):
        """
        Compute the Keplerian angular frequency at the NS surface

        """
        self.OmegaKep = (self.gravconst * self.NS_mass / self.NS_radius**3)**0.5
        self.OmegaKep_units = "s^-1"

    def Eval_viscous_time(self):
        """
        Compute the viscous timescale of the disk

        """
        #####################################################
        ## Inconsistent prescription used in Gompertz 2014...
        #####################################################
        #self.viscous_time = self.DISK_radius**2
        #self.DISK_cs = 1e7
        #self.viscous_time/= (3. * self.DISK_alpha * self.DISK_cs * self.DISK_radius)

        #####################################################
        ## More consistent definition of the viscous time....
        #####################################################
        H = self.DISK_radius * self.DISK_aspect_ratio
        cs = H*self.OmegaKep*(self.NS_radius/self.DISK_radius)**1.5
        self.viscous_time = self.DISK_radius**2 / (3. * self.DISK_alpha * cs * H)

        ######################
        ## don't forget units
        ######################
        self.viscous_time_units = "s"
        
    def Eval_Mdot0(self):
        """
        Compute the initial mass accretion rate
        (See eq (3) of King and Ritter 1998)

        """
        self.Mdot0 = self.DISK_mass0/self.viscous_time
        self.Mdot0_units = "g/s"
        
    def Eval_Omega0(self,verbose):
        """ 
        Set the neutron star initial angular frequency.

        If the neutron star period is not defined (==np.inf),
        it uses the critical value to avoid bar-mode instability.

        see beta = T/|W| parameter in
        Gompertz et al. 2014, MNRAS 438, 240-250 ; eq. (10)

        """
        if self.NS_period==np.inf:
            self.Omega0 = np.sqrt(2*self.NS_critical_beta*self.E_bind()/self.EOS_I)
            P0 = 2*np.pi/self.Omega0
            if verbose:
                print ('Setting Omega0 automatically\nin self.Eval_Omega0()')
                print ('Initial period = %.1e s'%P0)
        else:
            self.Omega0 = 2*np.pi/self.NS_period

        self.Omega0_units = "s^-1"


    def Eval_Tc(self):
        """
        Set the critical time Tc
        eq. (5) of Zhang & Meszaros (2001)

        """
        ##################
        ### temporary fix
        ##################
        self.q=-2 ## assume dipole injection
        self.q_units = ''

        prefac = (self.AG_alpha+self.q+1)
        term2 = prefac*(self.AG_Eimp/(self.L_em0*self.AG_T0))**(1./prefac)
        
        self.Tc = self.AG_T0*max(1,term2)
        self.Tc_units = 's'

    def Eval_T_em(self):
        """
        Compute the dipole spin-down time
        eq. (6) of Zhang & Meszaros (2001)

        Set Attribute:
                time_spindown

        """
        num = 3*self.lightspeed**3*self.EOS_I
        den = self.NS_B**2*self.NS_radius**6*self.Omega0**2

        self.time_spindown = num/den
        self.time_spindown_units = 's'
        
    def Eval_L_em0(self):
        """
        Set the plateau luminosity
        eq. (8) of Zhang & Meszaros (2001)

        """
        num = self.EOS_I*self.Omega0**2
        den = 2*self.T_em

        self.L_em0 = num/den
        self.L_em0_units = 'ergs/s'

    def Eval_critical_angular_velocity(self):
        """
        Sun, Zhang & Gao (2017)
        eq. (25)

        NS collapse for Omega < Omega_c (P>Pc)

        Rem: assume constant NS mass

        """
        num = self.NS_mass - self.EOS_Mtov

        if num<0:
            ## then NS always stable
            self.critical_period = -np.inf
            self.Omega_c = -np.inf

        else:
            den = self.EOS_alpha * self.EOS_Mtov
            self.critical_period = (num/den)**(1./self.EOS_beta)
            self.Omega_c = 2*np.pi/self.critical_period

        self.critical_period_units = 's'

    def Eval_T_tau(self,T):
        """
        Compute the time when the ejecta become optically thin
        """
        where_ejecta_thin = self.tau<=1
        i = np.argmax(where_ejecta_thin)
        if i > 0:
            self.time_opacity = T[i]
        else:
            self.time_opacity = -1

        self.time_opacity_units = 's'


    def Eval_T_col(self,T):
        """
        Compute the time when the supramassive NS collapses to a BH
        """
        where_NS_is_unstable = self.Omega < self.Omega_c
        i = np.argmax(where_NS_is_unstable)
        if i > 0:
            self.time_collapse = T[i]
        else:
            self.time_collapse = -np.inf

        self.time_collapse_units = 's'

    ##########################################################
    ### Functions computing time-dependent derived quantities:
    ### Characteristic radii
    ### Torques
    ### Rotational and gravitational energy
    ### Accretion rate
    ##########################################################
    def LC_radius(self,Omega):
        """
        Light cylinder radius (for a given NS rotation)

        """
        out = self.lightspeed/Omega

        return np.ascontiguousarray(out)

    def Magnetospheric_radius(self,T,Omega):
        """
        Magnetospheric radius 

        """
        Mdot = self.Accretion_rate(T)
        r_lc = self.LC_radius(Omega)
        out = self.mu**(4./7) * (self.gravconst*self.NS_mass)**(-1./7) * Mdot**(-2./7)

        mask = out > 0.999*r_lc
        out[mask] = 0.999*r_lc[mask]

        return out

    def Corotation_radius(self,Omega):
        """
        Corotation radius (for a given NS mass and spin)

        """
        out = (self.gravconst * self.NS_mass/ Omega**2)**(1./3)
        return out

    def E_rot(self,Omega):
        """ 
        Rotational energy of the NS

        """
        out=0.5*self.EOS_I*Omega**2
        return out

    def E_bind(self):
        """
        Binding energy of the NS
        Prescription from Lattimer and Prakash (2001)

        """
        num = self.gravconst*self.NS_mass
        den = self.NS_radius*self.lightspeed**2-0.5*self.gravconst*self.NS_mass
        out = 0.6*self.NS_mass*self.lightspeed**2*num/den
        return out
        
    def Accretion_rate(self,T):
        """
        Accretion rate on the NS
        Eq. (13) from Zhang and Meszaros 2001

        """
        out = self.Mdot0 * np.exp(-T / self.viscous_time)

        ## set a minimum value to avoid
        ## crashing during time integration
        mdot_floor=1e-10

        ## check with a mask
        out = np.ascontiguousarray(out)
        out[out<mdot_floor] = mdot_floor

        return out

    def Torque_spindown(self,T,Omega):
        """
        Dipole spindown torque. Eq (8) of Zhang and Meszaros 2001

        """
        ################################################################
        ## Gompertz uses the disk's alfven radius
        ## in the Bucciantini prescription,
        ## but it should actually be the alfven radius of the NS wind...
        ################################################################
        #r_lc  = self.LC_radius(Omega)
        #r_mag = self.Magnetospheric_radius(T,Omega)

        ###################################
        ## Eq (2) of Bucciantini et al 2006
        ###################################
        #out = - 2./3. * self.mu**2 * Omega**3 / self.lightspeed**3 * (r_lc/r_mag)**3

        ############################################
        ## Standard dipole spindown, no wind or disk
        ############################################
        out = - 1./6. * self.mu**2 * Omega**3 / self.lightspeed**3
        out=np.ascontiguousarray(out)

        #########################
        ## check NS stability
        #########################
        where_NS_is_unstable = Omega < self.Omega_c
        if np.any(where_NS_is_unstable):
            out[where_NS_is_unstable] = 0.
            warnings.warn('NS collapsed')

        return out

    def Torque_accretion(self,T,Omega):
        """
        Accretion torque, taking into account the propeller model
        Eq (6-7) of Gompertz et al. 2014

        """
        Mdot=self.Accretion_rate(T)

        ## Warning :
        ## radius of different types (array & float)
        r_lc = self.LC_radius(Omega)
        r_mag = self.Magnetospheric_radius(T,Omega)
        r_corot = self.Corotation_radius(Omega)

        fastness = (r_mag / r_corot)**1.5

        ## Eq. (6)
        out = (1. - fastness) * (self.gravconst * self.NS_mass * r_mag)**0.5 * Mdot

        ## Eq. (7)
        mask = r_mag<=self.NS_radius
        out[mask] = ((1. - Omega/self.OmegaKep) * (self.gravconst*self.NS_mass*r_mag)**0.5 * Mdot)[mask]

        ###############################################
        ## Check for inhibition by bar-mode instability
        ## with beta = T/|W| parameter (Gompertz 2014)
        ###############################################
        beta = self.E_rot(Omega)/abs(self.E_bind())
        out[beta>self.NS_critical_beta] = 0.

        #########################
        ## check NS stability
        #########################
        where_NS_is_unstable = Omega < self.Omega_c
        if np.any(where_NS_is_unstable):
            out[where_NS_is_unstable] = 0.
            warnings.warn('NS collapsed')

        return out

    ###############################################
    ### Functions computing the time derivatives of
    ### NS spin and Ejecta-related quantities:
    ### Omega
    ### co_Time
    ### Gamma
    ### co_Eint
    ### co_Volume
    ### Radius
    ###############################################
    def Omega_dot(self,Omega,T):
        """
        Time derivative of the NS spin used in the propeller model

        """
        r_lc = self.LC_radius(Omega)
        r_mag = self.Magnetospheric_radius(T,Omega)
        r_corot = self.Corotation_radius(Omega)

        Ndip = self.Torque_spindown(T,Omega)
        Nacc = self.Torque_accretion(T,Omega)

        out = (Ndip + Nacc)/self.EOS_I

        return np.ascontiguousarray(out)

    def co_Time_dot(self,Gamma):
        """
        return the derivative of the time (t')
        in the co-moving reference frame

        """
        out = self.Doppler_factor(Gamma)

        return np.ascontiguousarray(out)

    def Gamma_dot(self, T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius):
        """
        Eq. (14) Sun et al. (2017)

        """
        ##########################
        ### intermediate variables
        ##########################
        Doppler = self.Doppler_factor(Gamma)
        beta = self.Beta(Gamma)

        L_dip   = self.Luminosity_dipole(Omega,T)
        L_prop  = self.Luminosity_propeller(Omega,T)
        L_radio = self.Luminosity_radioactivity(co_Time,Gamma)
        L_elect = self.Luminosity_electrons(co_Eint,Gamma,co_Volume,Radius)

        L1 = L_dip + L_prop + L_radio - L_elect
        L2 = self.EJECTA_heating_efficiency*(L_dip + L_prop) + L_radio - L_elect

        ##########
        ### output
        ##########
        gdot = L1 - Gamma/Doppler * L2

        gdot+= Gamma*Doppler * co_Eint/(3*co_Volume) * 4*np.pi*beta*self.lightspeed*Radius**2

        gdot/=(self.EJECTA_mass*self.lightspeed**2 + co_Eint)

        return np.ascontiguousarray(gdot)

    def co_Eint_dot(self, T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius):
        """
        Eq. (15) Sun et al. (2017)

        """
        ##########################
        ### intermediate variables
        ##########################
        Doppler = self.Doppler_factor(Gamma)
        beta = self.Beta(Gamma)

        L_dip   = self.Luminosity_dipole(Omega,T)
        L_prop  = self.Luminosity_propeller(Omega,T)
        L_radio = self.Luminosity_radioactivity(co_Time,Gamma)
        L_elect = self.Luminosity_electrons(co_Eint,Gamma,co_Volume,Radius)

        L2 = self.EJECTA_heating_efficiency*(L_dip + L_prop) + L_radio- L_elect

        ##########
        ### output
        ##########
        Edot = 1/Doppler**2 * L2 - co_Eint/(3*co_Volume) * 4*np.pi*beta*self.lightspeed*Radius**2

        Edot*= Doppler

        return np.ascontiguousarray(Edot)

    def co_Volume_dot(self,Gamma,Radius):

        vdot = self.Doppler_factor(Gamma)*4*np.pi*self.lightspeed
        vdot*= Radius**2*self.Beta(Gamma)

        return np.ascontiguousarray(vdot)

    def Radius_dot(self,Gamma):

        beta = self.Beta(Gamma)
        rdot = beta*self.lightspeed/(1.-beta)
        return np.ascontiguousarray(rdot)

    ############################################
    ### Methods related to the time integration:
    ### initial conditions, RHS, integration
    ############################################
    def Initial_conditions(self):
        IC = (self.Omega0,
              self.EJECTA_co_Time0,
              self.EJECTA_Gamma0,
              self.EJECTA_co_Eint0,
              self.EJECTA_co_Volume0,
              self.EJECTA_radius0)
        return IC

    def Build_RHS(self, Y, T):
        """
        This function computes the time derivatives
        and is aimed to be passed to scipy.odeint()

        This determines its signature:

        Y : the unknowns

        T : time

        """
        ####################################
        ## expand variables
        ## these will be of numpy.float type
        ####################################
        Omega, co_Time, Gamma, co_Eint, co_Volume, Radius = Y

        ######################################
        ## compute each RHS
        ## the *_dot() methods return 1D array
        ######################################
        Omega_dot = self.Omega_dot(Omega,T)

        co_Time_dot = self.co_Time_dot(Gamma)

        Gamma_dot = self.Gamma_dot(T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius)

        co_Eint_dot = self.co_Eint_dot(T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius)

        co_Volume_dot = self.co_Volume_dot(Gamma,Radius)

        Radius_dot = self.Radius_dot(Gamma)

        ##########################################################
        ## repack and return
        ## the odeint routine expect a tuple of float/1D array
        ## similar to the initial conditions,
        ## or when we first expand the Y vector above
        ##
        ## That's the reason to enforce this type with the float()
        ## function !
        ##
        ## Rem: be careful to the order of the variables !
        ##########################################################
        out = (float(Omega_dot), float(co_Time_dot),
               float(Gamma_dot), float(co_Eint_dot),
               float(co_Volume_dot), float(Radius_dot))
        return out

    def Time_integration(self,time):
        """
        odeint wrapper

        """
        Y0 = self.Initial_conditions()
        sol = odeint(self.Build_RHS, Y0, time)

        (self.Omega, self.co_Time, self.Gamma, self.co_Eint,
        self.co_Volume, self.Radius) = sol.T


    ##################################
    ### Functions computing individual 
    ### luminosity components 
    ##################################
    def Luminosity_dipole(self,Omega,T):
        """
        Dipole spindown luminosity, for a general 
        time evolution of the NS angular velocity
        """
        Ndip = self.Torque_spindown(T,Omega)
        ldip = - Ndip * Omega
        #ldip*= -self.NS_eta_dip

        return ldip

    def Luminosity_propeller(self,Omega,T):
        """
        Propeller luminosity, taking into account
        positive and negative torques due to the 
        interaction with the accretion disk
        From Gompertz et al. (2014)
        """

        ### intermediate variables
        Mdot = self.Accretion_rate(T)
        Nacc = self.Torque_accretion(T,Omega)
        rmag = self.Magnetospheric_radius(T,Omega)

        ### output
        lprop = - Nacc*Omega - self.gravconst*self.NS_mass*Mdot/rmag
        lprop[lprop<0.] = 0.

        return lprop

    def Luminosity_radioactivity(self,cotime,Gamma):
        """
        Eq. (16) Sun et al. (2017)

        """
        Doppler = self.Doppler_factor(Gamma)
        prefactor = Doppler**2 * 4e49*self.EJECTA_mass/1e-2/self.Msun

        out = (0.5 - 1./np.pi*np.arctan((cotime-self.EJECTA_co_T0)/self.EJECTA_co_TSIGMA))**1.3
        out*=prefactor
        
        return out

    def Luminosity_electrons(self,Eint,Gamma,Volume,Radius):
        """
        Eq. (20) Sun (2017)

        """
        Doppler = self.Doppler_factor(Gamma)
        tau = self.Optical_depth(Gamma,Volume,Radius)

        where_ejecta_thin = tau<=1

        out = Doppler**2 * Eint*self.lightspeed*Gamma/(tau*Radius)

        out[where_ejecta_thin]*= tau[where_ejecta_thin]

        return out
    def Luminosity_AG(self,T):
        """
        Loss function
        Limp * T**(-alpha)
        Deprecated, not used in the free/trapped model
        """
        out = self.AG_Eimp * (T/self.AG_T0)**(-self.AG_alpha)
        return out

    def Luminosity_EM(self,T):
        """
        Analytic dipole spindown luminosity as a function of time 
        Eq. (7) of Zhang & Meszaros (2001)
        Deprecated, not used in the free/trapped model
        """
        out = self.NS_eta_dip*self.L_em0/(1.+T/self.T_em)**2
        return out

    ##############################################
    ## Function definitions for the trapped zone
    ##############################################
    def Beta(self,Gamma):
        """
        Gamma : Lorentz factor

        """
        return (1-Gamma**(-2))**0.5

    def Doppler_factor(self,Gamma):
        out = Gamma*(1. - self.Beta(Gamma)*np.cos(self.EJECTA_theta))
        return 1./out

    def Optical_depth(self,Gamma,Volume,Radius):

        out = (self.EJECTA_mass/Volume)*(Radius/Gamma)
        out*= self.EJECTA_opacity

        return np.ascontiguousarray(out)


    def Temperature(self,Gamma,Eint,Volume,Radius):
        """
        Black-Body temperature (Sun et al. 2017)
        """
        tau = self.Optical_depth(Gamma,Volume,Radius)
        where_ejecta_thin = tau<=1

        out = (Eint / self.radiation_const / Volume / tau)**(0.25)
        
        out[where_ejecta_thin] *= tau[where_ejecta_thin]**(0.25)

        return out

    #############################################
    ### Lightcurve definitions
    ### (contribution from dipole,
    ### accretion + propeller, radiative losses)
    ### X-Ray Lightcurves from free/trapped zones
    #############################################
    def Eval_LX_free(self,T):
        """
        X-Ray luminosity from dipole spindown and propeller
        """
        self.L_dip = self.Luminosity_dipole(self.Omega,T)
        self.L_prop = self.Luminosity_propeller(self.Omega,T)
        self.LX_free = (self.NS_eta_dip    * self.L_dip + 
                        self.DISK_eta_prop * self.L_prop)  
        
        self.L_dip_units   = 'ergs/s'
        self.L_prop_units  = 'ergs/s'
        self.LX_free_units = 'ergs/s'

    def Integrand_blackbody(self,nu):
        """
        define the function we want to integrate
        over a frequency range

        Eq. (22) (Sun et al. 2017)

        Parameter:
        ----------

        nu : 1D array (dtype=float)
                frequency range in Hz

        Returns:
        --------

        out : 2D array of size (nu,time)

        """
        out = np.empty((nu.size,self.t_num))

        ##################
        ### precalculation
        ##################
        Doppler = self.Doppler_factor(self.Gamma)
        Temp = self.Temperature(self.Gamma,self.co_Eint,
                                self.co_Volume,self.Radius)

        prefactor = 8. * (np.pi * Doppler * self.Radius)**2
        prefactor/= self.hPlanck**3 * self.lightspeed**2

        ########################
        ## loop over frequencies
        ########################
        for i,anu in enumerate(nu):
            num = (self.hPlanck*anu/Doppler)**4
            den = np.exp(self.hPlanck*anu/(Doppler*self.kBoltzmann*Temp)) - 1.
            out[i] = prefactor*(num/den) / anu

        return out

    def Integrate_blackbody(self,keV_min,keV_max,samples=1000):
        """
        Numerical integration of black body spectrum
        on a frequency range.

        It uses numpy.trapz()

        Parameters
        ----------

        keV_min, keV_max : float
                bound of the frequency range

        samples : int
                number of sample points for integration

        """
        nu_min = keV_min*1e3*self.Ev_to_Hz
        nu_max = keV_max*1e3*self.Ev_to_Hz

        freqs = np.linspace(nu_min,nu_max,samples)
        data = self.Integrand_blackbody(freqs)

        #### integration
        out = np.trapz(y=data, x=freqs, axis=0)
        return out

    def Eval_LX_trap(self):
        """
        X-Ray luminosity from trapped zone
        (Sun et al. 2017)
        """
        tau = self.Optical_depth(self.Gamma,self.co_Volume,self.Radius)
        L_wind = np.exp(-tau) * self.LX_free

        ####################################
        #################
        ## Sun model
        #################
        # nu = 1.e3 * self.Ev_to_Hz
        # Doppler = self.Doppler_factor(self.Gamma)
        # Temp = self.Temperature(self.Gamma,self.co_Eint,
        #                         self.co_Volume,self.Radius)
        # prefactor = 8. * (np.pi * Doppler * self.Radius)**2 
        # prefactor/= self.hPlanck**3 * self.lightspeed**2
        # num = (self.hPlanck * nu / Doppler)**4
        # den = np.exp(self.hPlanck * nu / (Doppler * self.kBoltzmann * Temp)) - 1.
        # L_bb = prefactor * num / den 
        #################
        ## or integration
        #################
        L_bb = self.Integrate_blackbody(keV_min=0.3,keV_max=6.)
        ####################################

        self.L_bb = L_bb
        self.LX_trap = L_wind + L_bb 
        self.LX_trap_units = 'ergs/s'
        

    #########################################
    ### Derived quantities used as diagnostic
    ### (characteristic radii, torques, etc.)
    ### Loaded as class members
    #########################################

    def Eval_radii(self,T):
        """
        Compute all the characteristic radii
        """
        Omega=self.Omega
        self.r_lc  = self.LC_radius(Omega)
        self.r_mag = self.Magnetospheric_radius(T,Omega)
        self.r_cor = self.Corotation_radius(Omega)

        self.r_lc_units  = 'cm'
        self.r_mag_units = 'cm'
        self.r_cor_units = 'cm'

    def Eval_torques(self,T):
        """
        Compute the various torques
        """
        Omega=self.Omega
        self.N_dip = self.Torque_spindown(T,Omega)
        self.N_acc = self.Torque_accretion(T,Omega)

        self.N_dip_units = 'erg'
        self.N_acc_units = 'erg'

    def Eval_diagnostic_outputs(self,T):
        """
        Compute various derived quantities
        """
        Omega = self.Omega
        self.Mdot = self.Accretion_rate(T)
        self.fast = (self.r_mag / self.r_cor)**1.5
        self.beta = self.E_rot(Omega)/abs(self.E_bind())
        self.tau  = self.Optical_depth(self.Gamma,
                                       self.co_Volume,self.Radius)
        self.Temp = self.Temperature(self.Gamma,self.co_Eint,
                                     self.co_Volume,self.Radius)
        self.L_elect = self.Luminosity_electrons(self.co_Eint,self.Gamma,
                                                 self.co_Volume,self.Radius)
        self.L_radio = self.Luminosity_radioactivity(self.co_Time,self.Gamma)

        self.Mdot_units = 'g/s'
        self.fast_units = ''
        self.beta_units = ''

    ########################################
    ### definition of the plotting functions
    ########################################
    
    def PlotLuminosity(self,T,
                       savefig=False,
                       filename='lightcurve.pdf'):
        """
        Plot lightcurves as a function of time

        Parameters:
        ----------

        T : array
                time

        savefig : boolean

        filename : string
                parameter to save the plot
                "path/name.format"
        """
        fig,ax = plt.subplots(2,1,figsize=(6,8))
#        fig,ax = plt.subplots()

        ax[0].loglog(T,self.LX_free,'r-',linewidth=3.0,label=r'$L_\text{x,free}$') 
        ax[0].loglog(T,self.LX_trap,'b--',linewidth=3.0,label=r'$L_\text{x,trap}$') 
        ax[0].loglog(T,self.L_dip,'k-.',label=r'$L_\text{dip}$')
        ax[0].loglog(T,self.L_prop,'g:',label=r'$L_\text{prop}$')

        ax[1].loglog(T,self.L_dip+self.L_prop,'r-',linewidth=3,label=r'$L_\text{sd}$',color='m')
        ax[1].loglog(T,self.L_radio,'b--',linewidth=3,label=r'$L_\text{radioactivity}$',color='k')
        ax[1].loglog(T,self.L_elect,'g:',linewidth=3,label=r'$L_\text{electrons}$',color='brown')
        ax[1].loglog(T,self.L_bb,'g.-',linewidth=3,label=r'$L_\text{bb}$',color='orange')
        ############
        ### labels
        ############
        ax[0].legend(fontsize='x-large',loc=0)
        ax[0].set_ylabel(r'Luminosity [erg/s]')
        ax[0].set_xlabel(r'time [s]')

        ax[1].legend(fontsize='x-large',loc=0)
        ax[1].set_ylabel(r'Luminosity [erg/s]')
        ax[1].set_xlabel(r'time [s]')
        ##############################
        ### set axis limits by hand...
        ##############################
        #ax.set(xlim=[1.,1e5],ylim=[1e42,1e52])

        plt.tight_layout()
        if savefig:
            ## implement specific filename generator
            ## here if needed
            plt.savefig(filename)
        return fig,ax

    def PlotRadii(self,T):
        """
        Plot characteristic radii as a function of time
        """
        fig,ax = plt.subplots()
        
        ### Plot of radii (magnetospheric, corotation, light-cylinder)
        ax.loglog(T,self.r_cor,label=r'$r_{corot}$')
        ax.loglog(T,self.r_lc,label=r'$r_{lc}$')
        if self.DISK_eta_prop > 0:
            ax.loglog(T,self.r_mag,label=r'$r_{mag}$')

        ############
        ### labels
        ############
        ax.legend(fontsize='x-large')
        ax.set_ylabel(r'radius [cm]')
        ax.set_xlabel(r'time [s]')

        ############################
        ### set axis limit by hand
        ### for the __main__ example
        ############################
        print ('Adjusting PlotRadii axes')
        ax.set(xlim=[1.,1e5],ylim=[1e6,1e9])

        plt.tight_layout()

        ### Plot of the beta parameter
        #ax.figure(3)
        #ax.loglog(T,self.beta)
        #ax.ylabel(r'$\beta$')
        #ax.xlabel(r'time [s]')
    
        ### vlines
        #plt.axvline(self.T_em,label=r'$T_{em}$',ls='--',color='gray')
        #plt.axvline(self.T0,label=r'$T_0$',ls='-',color='gray')
        #plt.axvline(self.Tc,label=r'$T_c$',ls='--',color='r')
        ### hlines
        #plt.axhline(self.L_em0,label=r'$L_{em,0}$',ls='--',color='gray')

    def WriteTable(self,
                   filename=None,
                   outputs = ('time','LX_free','LX_trap'),
                   myformat='fits',
                   overwrite=True,
                   **kwargs):
        """
        write some outputs as columns in a file 

        Parameter:
        ----------

        filename : string or None
                like 'path/filename.ext'
                if None, filename = 'tag.format'

        outputs : tuple of string
                attributes to write in different columns

        format : string
                astropy supported data format

        overwrite : boolean
                forces to overwrite an existing file

        **kwargs : 
                other astropy.Table.write() 
                keyword arguments

        """
        if filename is None:
            filename = '.'.join((self.tag,myformat))
        ##########################################
        ## A - short way
        ##
        ## input parameters treated as metadatas
        ## but write a file without a 'primaryHDU'
        ## still missing units
        ##########################################
        datas = [getattr(self,name) for name in outputs]
        table = Table(datas, names=outputs, meta=self.parameters)
        table.write(filename,
                    format=myformat,
                    overwrite=overwrite,**kwargs)

        # #######################
        # ## B - long way
        # #######################
        # ## writing FITS headers
        # #######################
        # hdr = fits.Header(self.parameters)

        # ###################################
        # ## creating columns
        # ## REM : not sure of the format 'K'
        # ###################################
        # cocos = [fits.Column(name=astring,
        #                      format='K',
        #                      array=getattr(self,astring))
        #          for astring in outputs]
        # hdu = fits.BinTableHDU.from_columns(cocos)

        # #######################
        # ## combine and write
        # #######################
        # primary_hdu = fits.PrimaryHDU(header=hdr)
        # hdul = fits.HDUList([primary_hdu, hdu])
        # hdul.writeto('%s.fits'%outfile,overwrite=overwrite)
        
if __name__=='__main__':

    ranges = {}
    ranges['NS_B'] = np.logspace(15,17,3)
    ranges['NS_mass'] = (1.4,)
    database = Generate_inputs(ranges)

    ## modelling of GRB 061006 with dipole + power law by Gompertz et al 2013
    GRB_061006 = {}
    GRB_061006['NS_period'] = 24.2e-3
    GRB_061006['NS_B'] = 14.1e15
    GRB_061006['NS_eta_dip']=1.
    GRB_061006['AG_alpha'] = 3.24
    GRB_061006['AG_T0'] = 200
    GRB_061006['DISK_mass0']=0.
    GRB_061006['DISK_eta_prop']=0.

    ## modelling of GRB 061006 with both propeller and dipole by Gompertz et al (2014)
    GRB_061006prop = {}
    GRB_061006prop['AG_T0'] = 4e0
    #GRB_061006prop['NS_period'] = 1.51e-3
    GRB_061006prop['NS_B'] = 1.48e15
    GRB_061006prop['AG_alpha'] = 5.0
#    GRB_061006prop['DISK_mass0']=2.01e-2
#    GRB_061006prop['DISK_radius']=400.e5
#    GRB_061006prop['NS_eta_dip']=0.05
#    GRB_061006prop['DISK_eta_prop']=0.4


    #GRBname = 'GRB061006'
    GRBname = 'GRB061006prop'

    if GRBname == 'GRB061006':
        grb = GRB(**GRB_061006,**EOS['GM1'])
        grb.PlotLuminosity(grb.time)
        grb.PlotRadii(grb.time)

    if GRBname == 'GRB061006prop':
        grb = GRB(**GRB_061006prop,**EOS['GM1'])
        fig,ax=grb.PlotLuminosity(grb.time)
        ## reset axes by hand
#        ax = plt.gca()
        for i in np.arange(ax.size):
            ax[i].set(xlim=[1.,1e8],ylim=[1e30,1e52])

        #grb.PlotRadii(grb.time)

        #grb.WriteTable()

    plt.show()
