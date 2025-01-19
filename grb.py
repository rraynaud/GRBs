#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Transient Lightcurve Modelling

References:

- `Zhang and Meszaros, 2001, ApJ, 552 <https://ui.adsabs.harvard.edu/abs/2001ApJ...552L..35Z/abstract>`_
- `Gompertz et al., 2014, MNRAS, 438 <https://ui.adsabs.harvard.edu/abs/2014MNRAS.438..240G/abstract>`_
- `Sun et al., 2017, ApJ, 835:7 <https://ui.adsabs.harvard.edu/abs/2017ApJ...835....7S/abstract>`_
- `Ai et al., 2018, ApJ, 860:57 <https://ui.adsabs.harvard.edu/abs/2018ApJ...860...57A/abstract>`_

"""
__author__="GRBs: Guilet, Raynaud, Bugli"
__email__ ="jerome.guilet@cea.fr ; raphael.raynaud@cea.fr ; matteo.bugli@cea.fr"
####################################
import os,sys
import numpy as np
import matplotlib as mpl
import itertools
from scipy.integrate import odeint
from scipy import interpolate 
from scipy import integrate
#from astropy.io import fits
from astropy.table import Table
import warnings
try:
    import magic
    mpl.rcParams.update(mpl.rcParamsDefault)
except ImportError:
    pass
import tidal_deformability
###########################
### plot parameters
###########################
mpl.rcParams['text.usetex'] = False
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
d_units['NS_ellipticity'] = ''
d_units['AG_T0'] = 's'
d_units['AG_Eimp'] = 'erg'
d_units['AG_alpha'] = ''
d_units['DISK_mass0'] = 'g'
d_units['DISK_radius'] = 'cm'
d_units['DISK_alpha'] = ''
d_units['DISK_aspect_ratio'] = ''
d_units['DISK_eta_prop'] = ''
d_units['DISK_fej'] = ''
d_units['EOS_name'] = ''
d_units['EOS_Mtov'] = 'g'
d_units['EOS_alpha'] = ''
d_units['EOS_beta'] = ''
d_units['EOS_I'] = 'g cm^2'
d_units['EOS_P0'] = 's'
d_units['EJECTA_mass'] = 'g'
d_units['EJECTA_dyn_Ye'] = ''
d_units['EJECTA_post_Ye'] = ''
d_units['EJECTA_heating_efficiency']=''
d_units['EJECTA_free_Gamma0'] = ''
d_units['EJECTA_trap_Gamma0'] = ''
d_units['EJECTA_free_co_T0'] = 's'
d_units['EJECTA_trap_co_T0'] = 's'
d_units['EJECTA_co_TSIGMA'] = 's'
d_units['EJECTA_free_co_Time0'] = 's'
d_units['EJECTA_trap_co_Time0'] = 's'
d_units['EJECTA_free_co_Eint0'] = 'erg'
d_units['EJECTA_trap_co_Eint0'] = 'erg'
d_units['EJECTA_free_co_Volume0'] = 'cm^3'
d_units['EJECTA_trap_co_Volume0'] = 'cm^3'
d_units['EJECTA_free_radius0']= 'cm'
d_units['EJECTA_trap_radius0']= 'cm'
d_units['EJECTA_theta']= 'rad'
d_units['tag']=''
d_units['Gompertz']=''
d_units['M1']='g'
d_units['M2']='g'
###########################################
## EOS database (Ai 2018, Table 1)
## to be completed
###########################################
EOS = {}
EOS['GM1']   = {'EOS_name' :'GM1',
                'EOS_Mtov' :2.37,
                'EOS_alpha':1.58e-10,
                'EOS_beta' :-2.84,
                'EOS_I'    :3.33e45,
                'NS_radius':12.05e5,
                'EOS_P0'   :0.72e-3}
EOS['Shen']  = {'EOS_name' :'Shen',
                'EOS_Mtov' :2.18,
                'EOS_alpha':4.678e-10,
                'EOS_beta' :-2.738,
                'EOS_I'    :4.675e45,
                'NS_radius':12.40e5,
                'EOS_P0'   :0.72e-3} #A TROUVER
EOS['BSk21'] = {'EOS_name' :'BSk21',
                'EOS_Mtov' :2.28,
                'EOS_alpha':2.81e-10,
                'EOS_beta' :-2.75,
                'EOS_I'    :4.37e45,
                'NS_radius':11.08e5,
                'EOS_P0'   :0.60e-3}
EOS['DD2']   = {'EOS_name' :'DD2',
                'EOS_Mtov' :2.42,
                'EOS_alpha':1.37e-10,
                'EOS_beta' :-2.88,
                'EOS_I'    :5.43e45,
                'NS_radius':11.89e5,
                'EOS_P0'   :0.65e-3}
EOS['DDME2'] = {'EOS_name' :'DDME2',
                'EOS_Mtov' :2.48,
                'EOS_alpha':1.966e-10,
                'EOS_beta' :-2.84,
                'EOS_I'    :5.85e45,
                'NS_radius':12.09e5,
                'EOS_P0'   :0.66e-3}
EOS['CDDM1'] = {'EOS_name' :'CDDM1',
                'EOS_Mtov' :2.21,
                'EOS_alpha':3.93e-16,
                'EOS_beta' :-5.0,
                'EOS_I'    :11.67e45,
                'NS_radius':13.99e5,
                'EOS_P0'   :0.83e-3}
EOS['CIDDM'] = {'EOS_name' :'CIDDM',
                'EOS_Mtov' :2.09,
                'EOS_alpha':2.58e-16,
                'EOS_beta' :-4.93,
                'EOS_I'    :8.645e45,
                'NS_radius':12.43e5,
                'EOS_P0'   :1.00e-3}
EOS['MIT2'] = {'EOS_name' :'MIT2',
                'EOS_Mtov' :2.08,
                'EOS_alpha':1.57e-15,
                'EOS_beta' :-4.58,
                'EOS_I'    :7.881e45,
                'NS_radius':11.48e5,
                'EOS_P0'   :0.71e-3}
EOS['MIT3'] = {'EOS_name' :'MIT3',
                'EOS_Mtov' :2.48,
                'EOS_alpha':3.35e-15,
                'EOS_beta' :-4.60,
                'EOS_I'    :13.43e45,
                'NS_radius':13.71e5,
                'EOS_P0'   :0.85e-3}
#To reproduce Gompertz's (2014) plots
# EOS['Basique'] = {'EOS_Mtov' :2.5,
#                 'EOS_alpha': 0, #pas de consideration d'effondrement en TN post merger
#                 'EOS_beta' :-1,
#                 'EOS_I'    :9.702e44, #I=0.35MR^2 avec M=1.4
#                 'NS_radius':10.0e5}
###########################################
def Generate_inputs(dico):
    """
    This fonctions is used to generate a grid of models

    Parameters
    ----------

    dico : dictionary
        form {key: array of values to explore}

    Returns
    -------
    list of dictionaries

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
    """This class defines a transient lightcurve model.  It implements a
    modified version of Sun et al. 2017, with Xray emission from a
    free zone and a trapped zone.  The spindown luminosity of the NS
    takes into account contributions from standard dipolar spindown
    and the propeller model.

    Main ouputs are stored in the following class members:

    - LX_free --> Free zone luminosity
    - LX_trap --> Trapped zone luminosity
    - L_dip   --> Dipolar spindown luminosity
    - L_prop  --> Propeller spindown luminosity

    Other time-dependent quantities are available, such as
    characteristic radii, torques, optical depth and temperature
    of the ejecta, etc.

    Time dependent variables

    - Omega (neutron star angular velocity)
    - Gamma (Lorentz factor)
    - Radius (radius of the ejecta)
    - co_Time : co-moving time
    - co_Eint (ejecta internal energy)
    - c_Volume (volume of the ejecta)

    Notes
    -----
    - The code works in CGS_Gaussian units
    - Methods of the form Eval_* set attributes
    - Methods of the form L_* return a timeserie (1D array)
    - Ejecta parameters for the trapped zone:
      *_co_* = quantity defined in the co-moving frame


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
                 NS_ellipticity=0.1,    #NS ellipticity for the GW spindown
                 AG_T0=10,
                 AG_Eimp=-np.inf,
                 AG_alpha=0,
                 DISK_mass0=1.e-2,
                 DISK_radius=5.e8, # 5000 km
                 DISK_alpha=0.1,   # disk viscosity parameter
                 DISK_aspect_ratio=0.3, # Aspect ratio H/R
                 DISK_eta_prop=0.4,
                 DISK_fej=0.4, # Proportion of the initial mass disk that goes in the ejecta
                 EOS_name='DD2',
                 EOS_Mtov=2.18, # Msun
                 EOS_alpha=4.678e-10,
                 EOS_beta=-2.738,
                 EOS_I=4.37e45,
                 EOS_P0=0.65e-3, #breakout period
                 EJECTA_mass=1.e-2,
                 #EJECTA_opacity=2, #to remove bc interpolation more acurate
                 EJECTA_heating_efficiency=0.5,
                 EJECTA_theta=0.,
                 EJECTA_free_Gamma0=1.2,
                 EJECTA_trap_Gamma0=1.,
                 EJECTA_free_co_T0=1.3, # eq. 15 Sun (2017)
                 EJECTA_trap_co_T0=1.3, # eq. 15 Sun (2017)
                 EJECTA_co_TSIGMA=0.11,
                 EJECTA_free_co_Time0=1.,
                 EJECTA_trap_co_Time0=1.,
                 EJECTA_free_co_Eint0=1e48,
                 EJECTA_trap_co_Eint0=1e48,
                 EJECTA_free_co_Volume0=4./3.*np.pi*1e24,
                 EJECTA_trap_co_Volume0=4./3.*np.pi*1e24,
                 EJECTA_free_radius0=1e10, #10^5 km
                 EJECTA_trap_radius0=1e10, #10^5 km
                 EJECTA_dyn_Ye = 0.19, #electron fraction for the dynamical ejecta
                 EJECTA_post_Ye = 0.22, #and for the post-merger ejecta
                 tag='notag',
                 verbose=True,
                 Gompertz = False,
                 M1 = 1.3, # Need the masses of the progenitors for the polynomial fits (opacity and disk mass)
                 M2 = 1.3): # for the reproduction of the results of Gompertz et al., 2014
        """
        Parameters
        ----------

        t_min : float
                start integration time

        t_max : float
                end integration time

        t_num : int
                number of time steps

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

        Example
        -------
        >>> # modelling of GRB 061006 with both
        >>> # propeller and dipole
        >>> # see Gompertz et al (2014)

        >>> import grb

        >>> GRB_061006 = {}
        >>> GRB_061006['AG_T0'] = 4e0
        >>> GRB_061006['AG_alpha'] = 5.0
        >>> GRB_061006['NS_B'] = 1.e13
        >>> GRB_061006['NS_mass'] = 2.4
        >>> GRB_061006['NS_eta_dip']=0.01
        >>> GRB_061006['DISK_eta_prop']=0.

        >>> mod = grb.GRB(**GRB_061006,**grb.EOS['DD2'])

        >>> mod.PlotLuminosity(mod.time)
        >>> mod.PlotRadii(mod.time)

        >>> # display the available EOS
        >>> grb.EOS.keys()

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
        self.Eval_L_em0()
        #self.Eval_Tc()
        self.Eval_magnetic_moment()
        self.Eval_OmegaKep()
        self.Eval_viscous_time()
        self.Eval_Mdot0()
        self.Eval_critical_angular_velocity()
        self.Eval_opacity()

        self.Eval_Disk_mass()
        self.Eval_Disk_mass0()
        self.Eval_Ejecta_PostMerger_mass()
        self.Eval_Ejecta_Dynamical_mass()
        #self.Eval_kappa_tanaka_interp_trap(self.EJECTA_Ye)
        #self.Eval_kappa_tanaka_interp_free(self.EJECTA_Free_Ye)
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
        self.Eval_L_pure_dipole()

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
        control_param.remove('EOS_name')

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

        .. math::
           \\tau_\\alpha  = \\frac{R_\\mathrm{disk}^2}{3 \\alpha c_s H}

        """
        #####################################################
        ## Inconsistent prescription used in Gompertz 2014...
        #####################################################
        if self.Gompertz == True:
            self.viscous_time = self.DISK_radius**2
            self.DISK_cs = 1e7
            self.viscous_time/= (3. * self.DISK_alpha * self.DISK_cs * self.DISK_radius)

        #####################################################
        ## More consistent definition of the viscous time....
        #####################################################
        else :
            H = self.DISK_radius * self.DISK_aspect_ratio
            cs = H*self.OmegaKep*(self.NS_radius/self.DISK_radius)**1.5
            #print('cs =', cs, 'cm/s (vs cs=1e7 Gompertz)')
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

        see :math:`\\beta = T/|W|` parameter in

        Gompertz et al. 2014, MNRAS 438, 240-250 ; eq. (10)

        """
        if self.NS_period==np.inf:
            self.Omega0 = np.sqrt(2*self.NS_critical_beta*self.E_bind()/self.EOS_I)
            P0 = 2*np.pi/self.Omega0
            if P0 < self.EOS_P0:
                P0 = self.EOS_P0
            if verbose:
                print ('Setting Omega0 automatically\nin self.Eval_Omega0()')
                #print ('Initial period = %.1e s'%P0)
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

#        self.L_em0 = self.Luminosity_EM(self.time)
        num = self.EOS_I*self.Omega0**2
        den = 2*self.time_spindown
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

        if num<=0:
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
        where_ejecta_thin = self.tau_trap<=1
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

    # def Eval_opacity(self, Ye):
    #     """
    #     Ye-Opacity relation Tanaka et al. 2019
    #     """
    #     Ye_values = np.array([0.10,0.20,0.25,0.35,0.40])
    #     tau_values = np.array([30,20,5,3,1])
    #     tau = interpolate.interp1d(Ye_values, tau_values, kind='linear')
    #     #return interpolate.interp1d(kappa[::-1],ye[::-1], kind='linear',fill_value='extrapolate')
    #     self.EJECTA_opacity = tau(Ye)
    #     self.EJECTA_opacity_units = 'cm^2/g'

    # def Eval_kappa_tanaka_interp_trap(self, Ye):
    #     """
    #     Ye-Opacity relation Tanaka et al. 2019
    #     """
    #     ye = np.array([0.01,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50])
    #     kappa = np.array([30.1,30.0,29.9,22.30,5.60,5.36,3.30,0.96,0.1])
    #     f = interpolate.interp1d(ye, kappa, kind='linear')
    #     #return interpolate.interp1d(kappa[::-1],ye[::-1], kind='linear',fill_value='extrapolate')
    #     self.EJECTA_opacity_tanaka_trap = f(Ye) #self.EJECTA_Ye
    #     print("Opacity trap: " + str(self.EJECTA_opacity_tanaka_trap))
    #     self.EJECTA_opacity_tanaka_trap_units = 'cm^2/g'

    # def Eval_kappa_tanaka_interp_free(self, Ye):
    #     """
    #     Ye-Opacity relation Tanaka et al. 2019
    #     """
    #     ye = np.array([0.01,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50])
    #     kappa = np.array([30.1,30.0,29.9,22.30,5.60,5.36,3.30,0.96,0.1])
    #     f = interpolate.interp1d(ye, kappa, kind='linear')
    #     #return interpolate.interp1d(kappa[::-1],ye[::-1], kind='linear',fill_value='extrapolate')
    #     self.EJECTA_opacity_tanaka_free = f(Ye) #self.EJECTA_Ye
    #     print("Opacity free: " + str(self.EJECTA_opacity_tanaka_free))
    #     self.EJECTA_opacity_tanaka_free_units = 'cm^2/g'

    def Eval_opacity(self): # Polynomial fit for the dynamical ejecta opacity
        tau = tidal_deformability.set_opacity(self.M1, self.M2, self.EOS_name)
        print('Opacity ' + str(tau))
        self.EJECTA_dyn_opacity = tau
        self.EJECTA_dyn_opacity_units = 'cm^2/g'

    def Eval_Disk_mass(self):
        Disk_mass = tidal_deformability.polynomial_fit_disk_mass(self.M1, self.M2, self.EOS_name) # ca c en masses solaires attention
        self.DISK_mass = Disk_mass
        print('Disk mass ' + str(Disk_mass))
        self.DISK_mass *= self.Msun
        self.DISK_mass_units = 'g'

    def Eval_Disk_mass0(self):
        self.DISK_mass0 = (1 - self.DISK_fej)*self.DISK_mass

    def Eval_Ejecta_PostMerger_mass(self): #The post-merger ejecta mass is defined as a fraction of the disk mass, so as not to create mass
        self.EJECTA_post_mass = self.DISK_fej*self.DISK_mass
        print('Post-merger ejecta mass ' + str(self.EJECTA_post_mass/self.Msun))
        #self.EJECTA_post_mass *= self.Msun #You already converted the disk mass in sun mass dumbass
        self.EJECTA_post_mass_units = 'g'

    def Eval_Ejecta_Dynamical_mass(self): #The dynamical ejecta mass is extracted directly from the polynomial fits
        self.EJECTA_dyn_mass = tidal_deformability.polynomial_fit_dynamical_ejecta(self.M1, self.M2, self.EOS_name)
        print('Dynamical ejecta mass ' + str(self.EJECTA_dyn_mass))
        self.EJECTA_dyn_mass *= self.Msun
        self.EJECTA_dyn_mass_units = 'g'

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
        out  = self.mu**(4./7) * (self.gravconst*self.NS_mass)**(-1./7) * Mdot**(-2./7)

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

    def Torque_dipole(self,T,Omega):
        """
        Dipole spindown torque. Eq (8) of Zhang and Meszaros 2001

        """
        ################################################################
        ## Gompertz uses the disk's alfven radius
        ## in the Bucciantini prescription,
        ## but it should actually be the alfven radius of the NS wind...
        ################################################################
        if self.Gompertz == True:
            r_mag = self.Magnetospheric_radius(T,Omega)
            r_lc  = self.LC_radius(Omega)
            out = - 2./3. * self.mu**2 * Omega**3 / self.lightspeed**3 * (r_lc/r_mag)**3

        ###################################
        ## Eq (2) of Bucciantini et al 2006
        ###################################
        #r_lc  = self.LC_radius(Omega)
        #mdot=1e-4/self.Msun         #To be changed to something that makes sense...
        #r_AL = (self.NS_B**2 * self.NS_radius**4 / mdot / Omega) **(1./3.)
        #out = - 2./3. * self.mu**2 * Omega**3 / self.lightspeed**3 * (r_lc/r_AL)**3

        ############################################
        ## Standard dipole spindown, no wind or disk
        ############################################
        else:
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

    def Torque_gravwaves(self,Omega):
        """
        Gravitational wave spindown torque (Zhang and Meszaros 2001)

        """
        out = - 32./5. * self.gravconst * self.EOS_I**2 * self.NS_ellipticity**2 * Omega**5 / self.lightspeed**5
        out=np.ascontiguousarray(out)
        out[Omega<1e4]=0
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

        Ndip  = self.Torque_dipole(T,Omega)
        Nacc  = self.Torque_accretion(T,Omega)
        Ngrav = self.Torque_gravwaves(Omega)

        out = (Ndip + Nacc + Ngrav)/self.EOS_I

        return np.ascontiguousarray(out)

    def co_Time_dot(self,Gamma):
        """
        return the derivative of the time (t')
        in the co-moving reference frame

        """
        out = self.Doppler_factor(Gamma)

        return np.ascontiguousarray(out)

    def Gamma_dot(self, T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius, free):
        """
        Eq. (14) Sun et al. (2017)

        """
        ##########################
        ### intermediate variables
        ##########################
        Doppler = self.Doppler_factor(Gamma)
        beta = self.Beta(Gamma)
        tau = self.Optical_depth(Gamma,co_Volume,Radius,free)
        #tau_trap = self.Optical_depth(Gamma,co_Volume,Radius,free=False)
        L_dip   = self.Luminosity_dipole(Omega,T)
        L_prop  = self.Luminosity_propeller(Omega,T)
        L_radio = self.Luminosity_radioactivity(co_Time,Gamma)
        L_elect = self.Luminosity_electrons(co_Eint,Gamma,co_Volume,Radius)

        sd_acceleration_injection = 1 - np.exp(-tau) #bc sd_wind_injection = np.exp(-tau)
        L1 = (1 - np.exp(-tau))*L_dip + L_prop + L_radio - L_elect

        #L2 = self.EJECTA_heating_efficiency*(L_dip + L_prop) + L_radio - L_elect
        L2 = self.EJECTA_heating_efficiency*((1 - np.exp(-tau))*L_dip + L_prop) + L_radio - L_elect

        ##########
        ### output
        ##########
        gdot = L1 - Gamma/Doppler * L2

        gdot+= Gamma*Doppler * co_Eint/(3*co_Volume) * 4*np.pi*beta*self.lightspeed*Radius**2

        if free:
            gdot/=(self.EJECTA_dyn_mass*self.lightspeed**2 + co_Eint)
        else:
            gdot/=((self.EJECTA_dyn_mass+self.EJECTA_dyn_mass)*self.lightspeed**2 + co_Eint)
        #gdot/=(self.EJECTA_dyn_mass*self.lightspeed**2 + co_Eint)

        return np.ascontiguousarray(gdot)

    # def Gamma_dot_free(self, T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius, free):
    #     """
    #     Eq. (14) Sun et al. (2017)

    #     """
    #     ##########################
    #     ### intermediate variables
    #     ##########################
    #     Doppler = self.Doppler_factor(Gamma)
    #     beta = self.Beta(Gamma)
    #     tau_free = self.Optical_depth(Gamma,co_Volume,Radius,free)
    #     #tau_trap = self.Optical_depth(Gamma,co_Volume,Radius,free=False)
    #     L_dip   = self.Luminosity_dipole(Omega,T)
    #     L_prop  = self.Luminosity_propeller(Omega,T)
    #     L_radio = self.Luminosity_radioactivity(co_Time,Gamma)
    #     L_elect = self.Luminosity_electrons(co_Eint,Gamma,co_Volume,Radius)

    #     sd_acceleration_injection = 1 - np.exp(-tau) #bc sd_wind_injection = np.exp(-tau)
    #     L1_free = (1 - np.exp(-tau))*L_dip + L_prop + L_radio - L_elect

    #     #L2 = self.EJECTA_heating_efficiency*(L_dip + L_prop) + L_radio - L_elect
    #     L2_free = self.EJECTA_heating_efficiency*((1 - np.exp(-tau))*L_dip + L_prop) + L_radio - L_elect

    #     ##########
    #     ### output
    #     ##########
    #     gdot_free = L1_free - Gamma/Doppler * L2_free

    #     gdot_free+= Gamma*Doppler * co_Eint/(3*co_Volume) * 4*np.pi*beta*self.lightspeed*Radius**2

    #     gdot_free/=(self.EJECTA_mass*self.lightspeed**2 + co_Eint)

    #     return np.ascontiguousarray(gdot)

    def co_Eint_dot(self, T, Omega, co_Time, Gamma, co_Eint, co_Volume, Radius, free):
        """
        Eq. (15) Sun et al. (2017)

        """
        ##########################
        ### intermediate variables
        ##########################
        Doppler = self.Doppler_factor(Gamma)
        beta = self.Beta(Gamma)
        tau = self.Optical_depth(Gamma,co_Volume,Radius,free)

        L_dip   = self.Luminosity_dipole(Omega,T)
        L_prop  = self.Luminosity_propeller(Omega,T)
        L_radio = self.Luminosity_radioactivity(co_Time,Gamma)
        L_elect = self.Luminosity_electrons(co_Eint,Gamma,co_Volume,Radius)

        #L2 = self.EJECTA_heating_efficiency*(L_dip + L_prop) + L_radio- L_elect
        L2 = self.EJECTA_heating_efficiency*((1 - np.exp(-tau))*L_dip + L_prop) + L_radio- L_elect


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
              self.EJECTA_free_co_Time0, self.EJECTA_trap_co_Time0,
              self.EJECTA_free_Gamma0, self.EJECTA_trap_Gamma0,
              self.EJECTA_free_co_Eint0, self.EJECTA_trap_co_Eint0,
              self.EJECTA_free_co_Volume0, self.EJECTA_trap_co_Volume0,
              self.EJECTA_free_radius0, self.EJECTA_trap_radius0)
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
        Omega, co_Time_free, co_Time_trap, Gamma_free, Gamma_trap, co_Eint_free, co_Eint_trap, co_Volume_free, co_Volume_trap, Radius_free, Radius_trap = Y

        ######################################
        ## compute each RHS
        ## the *_dot() methods return 1D array
        ######################################
        Omega_dot = self.Omega_dot(Omega,T)

        co_Time_dot_free = self.co_Time_dot(Gamma_free)
        co_Time_dot_trap = self.co_Time_dot(Gamma_trap)

        Gamma_dot_free = self.Gamma_dot(T, Omega, co_Time_free, Gamma_free, co_Eint_free, co_Volume_free, Radius_free, free=True)
        Gamma_dot_trap = self.Gamma_dot(T, Omega, co_Time_trap, Gamma_trap, co_Eint_trap, co_Volume_trap, Radius_trap, free=False)

        co_Eint_dot_free = self.co_Eint_dot(T, Omega, co_Time_free, Gamma_free, co_Eint_free, co_Volume_free, Radius_free, free=True)
        co_Eint_dot_trap = self.co_Eint_dot(T, Omega, co_Time_trap, Gamma_trap, co_Eint_trap, co_Volume_trap, Radius_trap, free=False)

        co_Volume_dot_free = self.co_Volume_dot(Gamma_free,Radius_free)
        co_Volume_dot_trap = self.co_Volume_dot(Gamma_trap,Radius_trap)

        Radius_dot_free = self.Radius_dot(Gamma_free)
        Radius_dot_trap = self.Radius_dot(Gamma_trap)

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
        out = (float(Omega_dot),
               float(co_Time_dot_free), float(co_Time_dot_trap),
               float(Gamma_dot_free), float(Gamma_dot_trap),
               float(co_Eint_dot_free), float(co_Eint_dot_trap),
               float(co_Volume_dot_free), float(co_Volume_dot_trap),
               float(Radius_dot_free), float(Radius_dot_trap))
        return out

    def Time_integration(self,time):
        """
        odeint wrapper

        """
        Y0 = self.Initial_conditions()

        sol = odeint(self.Build_RHS, Y0, time)

        (self.Omega, self.co_Time_free, self.co_Time_trap, self.Gamma_free, self.Gamma_trap, self.co_Eint_free, self.co_Eint_trap,
        self.co_Volume_free, self.co_Volume_trap, self.Radius_free, self.Radius_trap) = sol.T


    ##################################
    ### Functions computing individual
    ### luminosity components
    ##################################

    def Luminosity_dipole(self,Omega,T):
        """
        Dipole spindown luminosity, for a general
        time evolution of the NS angular velocity
        """
        Ndip = self.Torque_dipole(T,Omega)
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
        prefactor = Doppler**2 * 4e49*self.EJECTA_dyn_mass/1e-2/self.Msun

        out = (0.5 - 1./np.pi*np.arctan((cotime-self.EJECTA_trap_co_T0)/self.EJECTA_co_TSIGMA))**1.3
        out*=prefactor

        return out

    def Luminosity_electrons(self,Eint,Gamma,Volume,Radius):
        """
        Eq. (20) Sun (2017)

        """
        free = False
        Doppler = self.Doppler_factor(Gamma)
        tau_trap = self.Optical_depth(Gamma,Volume,Radius,free)

        where_ejecta_thin = tau_trap<=1

        out = Doppler**2 * Eint*self.lightspeed*Gamma/(tau_trap*Radius)

        out[where_ejecta_thin]*= tau_trap[where_ejecta_thin]

        return out
        
    def Luminosity_AG(self,T):
        """
        Loss function
        Limp * T**(-alpha)

        Notes
        -----
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
        out = self.NS_eta_dip*self.L_em0/(1.+T/self.time_spindown)**2
        return out

    ##############################################
    ## Function definitions for the trapped zone
    ##############################################
    def Beta(self,Gamma):
        """
        Returns:
            float:  Lorentz factor

        """
        return (1-Gamma**(-2))**0.5

    def Doppler_factor(self,Gamma):
        """
        Returns:
            float:  Doppler factor
        """
        out = Gamma*(1. - self.Beta(Gamma)*np.cos(self.EJECTA_theta))
        return 1./out

    # def Optical_depth_free(self,Gamma,Volume,Radius):
    #     """
    #         2 components of the ejecta contribute to the opacity: the dynamical ejecta and post-merger ejecta
    #     """

    #     # out = (self.EJECTA_mass/Volume)*(Radius/Gamma)
    #     # out*= self.EJECTA_opacity

    #     #out*= self.EJECTA_opacity_tanaka
    #     out = ((self.EJECTA_mass/10)/Volume)*(Radius/Gamma) #Estimated mass fraction to go in free 1/10
    #     out*= self.EJECTA_opacity #* 26.56/360
    #     # mass_dyn =
    #     # mass_post =
    #     # opacity_tot = opacity_dyn * mass_dyn + opacity_post * mass_post
    #     # opacity_tot/= mass_dyn + mass_post # The resulting opacity is the weighted mean of the dynamical and post-merger ejecta

    #     # out = ((mass_dyn + mass_post)/Volume)*(Radius/Gamma)
    #     # out*= opacity_tot

    #     return np.ascontiguousarray(out)

    def Optical_depth(self,Gamma,Volume,Radius,free):
        """
            2 components of the ejecta contribute to the opacity: the dynamical ejecta (trapped only) and post-merger ejecta
        """

        # out = (self.EJECTA_mass/Volume)*(Radius/Gamma)
        # #out*= self.EJECTA_opacity_tanaka
        # out*= self.EJECTA_opacity#_tanaka

        # if free == True:
        #     out = ((self.EJECTA_mass/10)/Volume)*(Radius/Gamma) #Estimated mass fraction to go in free 1/10
        #     out*= self.EJECTA_opacity_tanaka_free
        # else:
        # out = (self.EJECTA_mass/Volume)*(Radius/Gamma)
        # out*= self.EJECTA_opacity

                # mass_dyn =
        # mass_post =
        if free == True: # In the free zone, only the dynamical ejecta is present, in 1/10 of the total dynamical ejecta mass
            out = (self.EJECTA_dyn_mass/Volume)*(Radius/Gamma)
            out*= self.EJECTA_dyn_opacity

        else:
            # opacity_dyn = Eval_opacity(Ye_dyn)
            self.EJECTA_post_opacity = tidal_deformability.opacity_interpolate(self.EJECTA_post_Ye)
            self.EJECTA_post_opacity_units = 'cm^2/g'
            # opacity_tot = self.EJECTA_dyn_opacity * self.EJECTA_dyn_mass + self.EJECTA_dyn_opacity * self.EJECTA_post_mass
            # opacity_tot/= self.EJECTA_dyn_mass + self.EJECTA_post_mass # The resulting opacity is the weighted mean of the dynamical and post-merger ejecta
            #print("Opacity free zone: " + str(self.EJECTA_dyn_opacity))
            opacity_tot = self.EJECTA_dyn_opacity * self.EJECTA_dyn_mass + self.EJECTA_post_opacity * self.EJECTA_post_mass
            opacity_tot/= self.EJECTA_dyn_mass + self.EJECTA_post_mass
            #print("Opacity trapped zone: " + str(opacity_tot))
            out = ((self.EJECTA_dyn_mass + self.EJECTA_post_mass)/Volume)*(Radius/Gamma)
            #out = (self.EJECTA_dyn_mass/Volume)*(Radius/Gamma)
            out*= opacity_tot


        return np.ascontiguousarray(out)


    # def Optical_depth_trap(self,Gamma,Volume,Radius):
    #     """
    #         2 components of the ejecta contribute to the opacity: the dynamical ejecta and post-merger ejecta
    #     """
    #     out = (self.EJECTA_mass/Volume)*(Radius/Gamma)
    #     out*= self.EJECTA_opacity * (1 - 26.56/360)

    #     return np.ascontiguousarray(out)


    def Temperature(self,Gamma,Eint,Volume,Radius):
        """
        Black-Body temperature (Sun et al. 2017)
        """
        free = False
        tau_trap = self.Optical_depth(Gamma,Volume,Radius,free)
        where_ejecta_thin = tau_trap<=1

        out = (Eint / self.radiation_const / Volume / tau_trap)**(0.25)

        out[where_ejecta_thin] *= tau_trap[where_ejecta_thin]**(0.25)

        return out

    #############################################
    ### Lightcurve definitions
    ### (contribution from dipole,
    ### accretion + propeller, radiative losses)
    ### X-Ray Lightcurves from free/trapped zones
    #############################################
    def Eval_L_pure_dipole(self):
        """
        X-Ray luminosity from pure dipole spindown
        (analytic formula from Zhang & Meszaros 2001)
        """
        self.L_pure_dip = self.Luminosity_EM(self.time)

        self.L_pure_dip_units   = 'ergs/s'

    def Eval_LX_free(self,T):
        """
        X-Ray luminosity from dipole spindown and propeller
        """
        free = True
        self.L_dip = self.Luminosity_dipole(self.Omega,T)
        self.L_prop = self.Luminosity_propeller(self.Omega,T)
        self.LX_free = (self.NS_eta_dip    * self.L_dip +
                        self.DISK_eta_prop * self.L_prop)
        tau_free = self.Optical_depth(self.Gamma_free,self.co_Volume_free,self.Radius_free,free) #* 26.56/360
        #print(tau)
        self.LX_free = np.exp(-tau_free) * self.LX_free

        self.L_dip_units   = 'ergs/s'
        self.L_prop_units  = 'ergs/s'
        self.LX_free_units = 'ergs/s'

    def Integrand_blackbody(self,nu):
        """
        define the function we want to integrate
        over a frequency range

        Eq. (22) (Sun et al. 2017)

        Parameters
        ----------

        nu : 1D array (dtype=float)
             frequency range in Hz

        Returns
        -------

        out : 2D array of size (nu,time)

        """
        out = np.empty((nu.size,self.t_num))

        ##################
        ### precalculation
        ##################
        Doppler = self.Doppler_factor(self.Gamma_trap)
        Temp = self.Temperature(self.Gamma_trap,self.co_Eint_trap,
                                self.co_Volume_trap,self.Radius_trap)

        prefactor = 8. * (np.pi * Doppler * self.Radius_trap)**2
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
        free = False
        tau_trap = self.Optical_depth(self.Gamma_trap,self.co_Volume_trap,self.Radius_trap,free)
        L_wind = np.exp(-tau_trap) * self.LX_free

        ####################################
        #################
        ## Sun model, no actual integration in frequency
        #################
        #nu = 1.e3 * self.Ev_to_Hz
        #Doppler = self.Doppler_factor(self.Gamma)
        #Temp = self.Temperature(self.Gamma,self.co_Eint,
        #                        self.co_Volume,self.Radius)
        #prefactor = 8. * (np.pi * Doppler * self.Radius)**2
        #prefactor/= self.hPlanck**3 * self.lightspeed**2
        #num = (self.hPlanck * nu / Doppler)**4
        #den = np.exp(self.hPlanck * nu / (Doppler * self.kBoltzmann * Temp)) - 1.
        #L_bb = prefactor * num / den
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
        self.N_dip  = self.Torque_dipole(T,Omega)
        self.N_acc  = self.Torque_accretion(T,Omega)
        self.N_grav = self.Torque_gravwaves(Omega)

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
        self.tau_free  = self.Optical_depth(self.Gamma_trap,
                                       self.co_Volume_free,self.Radius_free,free=True)
        self.tau_trap  = self.Optical_depth(self.Gamma_free,
                                       self.co_Volume_trap,self.Radius_trap,free=False)
        self.Temp = self.Temperature(self.Gamma_trap,self.co_Eint_trap,
                                     self.co_Volume_trap,self.Radius_trap)
        self.L_elect = self.Luminosity_electrons(self.co_Eint_trap,self.Gamma_trap,
                                                 self.co_Volume_trap,self.Radius_trap)
        self.L_radio = self.Luminosity_radioactivity(self.co_Time_trap,self.Gamma_trap)

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

        Parameters
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

        ax[0].loglog(T,self.LX_free,'r-',linewidth=3.0,label=r'$L_{\rm x,free}$')
        ax[0].loglog(T,self.LX_trap,'b--',linewidth=3.0,label=r'$L_{\rm x,trap}$')
        #ax[0].loglog(T,self.L_dip,'k-.',label=r'$L_{\rm dip}$')
        ax[0].loglog(T,self.L_prop,'g:',label=r'$L_{\rm prop}$')
        #ax[0].loglog(T,self.L_pure_dip,'y-',label=r'$L_{\rm dip,th}$')

        ax[1].loglog(T,self.L_dip+self.L_prop,ls='-',linewidth=3,label=r'$L_{\rm sd}$',color='m')
        ax[1].loglog(T,self.L_radio,ls='--',linewidth=3,label=r'$L_{\rm radioactivity}$',color='k')
        ax[1].loglog(T,self.L_elect,ls=':',linewidth=3,label=r'$L_{\rm electrons}$',color='brown')
        ax[1].loglog(T,self.L_bb,ls='-.',linewidth=3,label=r'$L_{\rm bb}$',color='orange')
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
        ax[1].set(xlim=[1.,1e5],ylim=[1e42,1e52])
        ax[0].set(xlim=[1.,1e5],ylim=[1e42,1e52])

        plt.tight_layout()
        if savefig:
            ## implement specific filename generator
            ## here if needed
            plt.savefig(filename, dpi=400)
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


    def PlotBeta(self, T,
                    savefig=False,
                    filename='beta.pdf'):

        ## Plot of the beta parameter
        #ax.figure(3)
        #beta_v = Beta()
        plt.loglog(T,self.Gamma_trap)
        plt.ylabel(r'$\beta$')
        plt.xlabel(r'time [s]')


        ## vlines
        # plt.axvline(self.T_em,label=r'$T_{em}$',ls='--',color='gray')
        # plt.axvline(self.T0,label=r'$T_0$',ls='-',color='gray')
        # plt.axvline(self.Tc,label=r'$T_c$',ls='--',color='r')
        ## hlines
        #plt.axhline(self.L_em0,label=r'$L_{em,0}$',ls='--',color='gray')
        plt.tight_layout()
        if savefig:
            ## implement specific filename generator
            ## here if needed
            plt.savefig(filename, dpi=400)

        plt.show()


    def Energy_kin_ejecta(self):
        E = (np.max(self.Gamma_free)-1) * self.EJECTA_dyn_mass * self.lightspeed**2
        print("Injected kinetic energy in ejecta: " + str(E) + "erg.")
        return E

    def Energy_reservoir(self):
        E = 1/2 * self.EOS_I * self.Omega0**2
        print("Energy reservoir: " + str(E) + "erg.")
        return E


    def Test_energies(self):

        def Luminosities(t0, tF, samples=1000):
            times = np.linspace(t0, tF, samples)

            Y0 = self.Initial_conditions()
            sol = odeint(self.Build_RHS, Y0, times)

            (iOmega, _, _, iGamma_free, iGamma_trap, _, _, iVolume_free, iVolume_trap, iRadius_free, iRadius_trap) = sol.T

            #LX_free
            L_dip = self.Luminosity_dipole(iOmega, times)
            L_prop = self.Luminosity_propeller(iOmega, times)
            LX_free = (self.NS_eta_dip * L_dip +
                            self.DISK_eta_prop * L_prop)
            tau_free = self.Optical_depth(iGamma_free,iVolume_free,iRadius_free,free=True)
            LX_free = np.exp(-tau_free) * LX_free
            #LX_trap
            tau_trap = self.Optical_depth(iGamma_trap,iVolume_trap,iRadius_trap,free=False)
            L_wind = np.exp(-tau_trap) * LX_free
            L_bb = self.Integrate_blackbody(keV_min=0.3,keV_max=6.,samples=samples)
            LX_trap = L_wind + L_bb

            L_tot = LX_free + LX_trap
            return L_tot

        t0 = 0
        tF = 10e3 #10e5
        y = Luminosities(t0, tF) #, limit=199
        Energy_tot = np.sum(y) * (tF-t0)/len(y)
        #Energy_tot_quad = integrate.quad(Luminosities, t0, tF)
        print("And integrated luminosities (to check the new I expression), which should be equal or lower: " + str(Energy_tot) + "erg.")
        #print("Please be also equal to: " + str(Energy_tot_quad) + "erg.")

        return Energy_tot




    def WriteTable(self,
                   filename=None,
                   outputs = ('time','LX_free','LX_trap'),
                   myformat='fits',
                   overwrite=True,
                   **kwargs):
        """
        write some outputs as columns in a file

        Parameters
        ----------

        filename : string or None
                like 'path/filename.ext'
                if None, filename = 'tag.format'

        outputs : tuple of string
                attributes to write in different columns

        format : string
                 astropy supported `data format <http://docs.astropy.org/en/stable/table/io.html>`_

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
    GRB_061006prop['AG_alpha'] = 5.0
    GRB_061006prop['NS_B'] = 1.e13
    GRB_061006prop['NS_mass'] = 2.4
#    GRB_061006prop['NS_period'] = 1.5e-3
    GRB_061006prop['NS_eta_dip']=0.01
    GRB_061006prop['DISK_eta_prop']=0.


#     #GRBname = 'GRB061006'
    #GRBname = 'GRB061006prop'

#     if GRBname == 'GRB061006':
#         grb = GRB(**GRB_061006,**EOS['GM1'])
#         grb.PlotLuminosity(grb.time)
#         grb.PlotRadii(grb.time)

    # if GRBname == 'GRB061006prop':
    #     grb = GRB(**GRB_061006prop,**EOS['DD2'])
    #     fig,ax=grb.PlotLuminosity(grb.time)
#         ## reset axes by hand
# #        ax = plt.gca()
# #        for i in np.arange(ax.size):
# #            ax[i].set(xlim=[1.,1e8],ylim=[1e30,1e52])

#         #grb.PlotRadii(grb.time)

#         #grb.WriteTable()

#     plt.show()



    GRB_base = {}
    GRB_base['NS_period'] = 1e-3
    GRB_base['NS_B'] = 1e15
    GRB_base['NS_mass'] = 2.43
    #GRB_base['EJECTA_Gamma0'] = 1.0
    # f_ej = 0.3
    # GRB_base['DISK_mass0'] = (1 - f_ej)*1.e-2
    # GRB_base['EJECTA_mass'] = f_ej*1.e-2



    #GRBname = 'GRB061006'
#     GRBname = 'GRB061006prop'
    #GRBname = 'GRB_test_I'

    grb = GRB(**GRB_base,**EOS['DD2'], t_num=1000)
    grb.PlotLuminosity(grb.time, savefig=True, filename='test_free/taufree_corrected_test_2taus.png')
    # plt.close()
    # grb.PlotBeta(grb.time, savefig=True, filename='test_tout/taufree.png')
    plt.show()
    grb.Energy_kin_ejecta()
    grb.Energy_reservoir()
    grb.Test_energies()
