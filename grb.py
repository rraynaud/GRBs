#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Transient Lightcurve Modelling

Based on Zhang & Meszaros (2001)
"""
__author__="GRBs: Guilet, Raynaud, Bugli"
__email__ ="jerome.guilet@cea.fr ; raphael.raynaud@cea.fr ; matteo.bugli@cea.fr"
####################################
import os,sys
import numpy as np
import matplotlib as mpl
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
###########################
class GRB(object):
    """This class defines a transient lightcurve model. 

    Notes
    -----
    The code works in CGS_Gaussian units
    Methods of the form Eval_* set attributes
    Methods of the form L_* return a timeserie (1D array)

    """
    def __init__(self,
                 t_min=0,
                 t_max=6,
                 t_num=200,
                 NS_B=1e15,
                 NS_mass=1.4,
                 NS_radius=1e6,
                 NS_P=1e-3,
                 NS_eta_dip=1,
                 T0=10,
                 Eimp=1,
                 alpha=0,
                 Mdisk=0.1,
                 Rdisk=500.0e5, # 500 km
                 alpha_disk=0.1, # disk viscosity parameter
                 cs=1.e7, # sound speed in the disk (100km/s)
                 eta_prop=1,
                 EoS_Mtov=2.18, # Msun
                 EoS_alpha=0.0766,
                 EoS_beta=-2.738,
                 tag='notag',
                 verbose=True):
        """
        Parameters (in cgs units, when not specified otherwise)
        ----------

        t_min, t_max, t_num : float, float, int
                define the integration time

        NS_B : float
                magnetar magnetic field
        
        NS_P : float
                magnetar period

        NS_mass : float
                magnetar mass (in units of solar mass)

        NS_radius : float
                magnetar radius

        NS_eta_dip : float
                dipole efficiency factor                 
        
        Mdisk : float
                disk mass (in units of solar mass)

        Rdisk : float
                disk radius

        alpha_disk : float
                disk viscosity parameter

        cs : float
                sound speed in the disk

        eta_prop : float
                propeller efficiency factor

        EoS_Mtov : float
                maximum mass of a NS with zero spin

        EoS_alpha : float
                phenomenological parameter used
                to compute the NS maximum mass

        EoS_beta : float
                similar to EoS_alpha

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
        del self.parameters['verbose']
        del self.parameters['__class__']
        del self.parameters['self']
        ###########################
        ### astrophysical constants
        ###########################
        self.lightspeed = 299792458e2 # cm/s
        self.Msun = 1.98855e33 # g
        ### gravitational constant
        self.gravconst = 6.67259e-8 # cm^3 g^-1 s^-2

        ###########################################
        ## input : set inputs and their dimensions
        ## NB: ALTERNATIVE use self.parameters and
        ## define a dictionnary to store the units
        ###########################################
        ## define time
        time = np.logspace(t_min,t_max,t_num)
        self.time = time
        self.time_units = 's'

        self.NS_eta_dip = NS_eta_dip
        self.NS_eta_dip_units = ''
        self.eta_prop = eta_prop
        self.eta_prop_units = ''
        self.Eimp = Eimp
        self.Eimp_units = 'erg'
        self.T0 = T0
        self.T0_units = 's'
        self.alpha = alpha
        self.alpha_units = '' 
        self.NS_B = NS_B
        self.NS_B_units = 'G'
        self.NS_P0 = NS_P ## 2 names for one var; not ideal
        self.NS_P0_units = 's'
        self.NS_mass = NS_mass * self.Msun
        self.NS_mass_units = 'g'
        self.NS_radius = NS_radius
        self.NS_radius_units = 'cm'
        self.Mdisk0 = Mdisk * self.Msun
        self.Mdisk0_units = 'g'
        self.Rdisk0 = Rdisk
        self.Rdisk0_units = 'cm'
        self.alpha_disk = alpha_disk
        self.alpha_disk_units = ''
        self.cs = cs
        self.cs_units = 'cm/s'
        self.tag = tag
        self.EoS_Mtov = EoS_Mtov * self.Msun
        self.EoS_Mtov_units = 'g'
        self.EoS_alpha = EoS_alpha
        self.EoS_alpha_units = ''
        self.EoS_beta = EoS_beta
        self.EoS_beta_units = ''
        ##################
        ### temporary fix
        ##################
        self.q=-2 ## assume dipole injection
        self.q_units = ''

        ######################
        ### derived quantities
        ######################
        self.Eval_MomentOfInertia()
        self.Eval_Omega0()
        self.Eval_T_em()
        self.Eval_L_em0()
        self.Eval_Tc()
        self.Eval_magnetic_moment()
        self.Eval_OmegaKep()
        self.Eval_viscous_time()
        self.Eval_Mdot0()
        self.Eval_critical_angular_velocity()
        ######################
        ### fine tuning
        ######################
        print ('Fine tuning ON...')
        self.Eimp = self.L_em0#*self.T0

        ######################
        ### Light curves
        ######################
        self.Eval_Omega(time)
        self.Eval_radii(time)
        self.Eval_torques(time)
        self.Eval_diag(time)
        self.Eval_L_rad(time)
        self.Eval_L_dip()
        self.Eval_L_prop(time)
        self.Eval_L_tot(time)

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
        control_param = ('NS_B','NS_P0','NS_radius','NS_mass','alpha','Eimp','T0',
                         'Mdisk0','Rdisk0','alpha_disk','cs',
                         'NS_eta_dip','eta_prop')
        derived_param = ('T_em','Tc','I','L_em0','mu',
                         'viscous_time','Mdot0','OmegaKep')
        ### for the layout column width
        lenun = max([len(getattr(self,afield+'_units'))
                     for afield in control_param+derived_param])
        lensy = max([len(afield)
                     for afield in control_param+derived_param])
        
        header = '{:-^%i}'%(lenun+lensy+2+8)
        ligne = '{:%i} {:8.2e} {:%i}'%(lensy,lenun)
        
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
    ########################################
    ### definition of the derived parameters
    ########################################
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
        self.I = norm * self.NS_mass*self.NS_radius**2
        self.I_units = 'g cm^2'

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
        self.viscous_time = self.Rdisk0**2
        self.viscous_time/= (3. * self.alpha_disk * self.cs * self.Rdisk0)

        #####################################################
        ## More consistent definition of the viscous time (?)
        #####################################################
        # cs = self.Rdisk0*self.OmegaKep/np.sqrt(2)*(self.R/self.Rdisk0)**1.5
        # self.viscous_time = self.Rdisk0**2 / (3. * self.alpha_disk * cs * self.Rdisk0)

        ######################
        ## don't forget units
        ######################
        self.viscous_time_units = "s"
        
    def Eval_Mdot0(self):
        """
        Compute the initial mass accretion rate
        (See eq (3) of King and Ritter 1998)

        """
        self.Mdot0 = self.Mdisk0/self.viscous_time
        self.Mdot0_units = "g/s"
        
    def Eval_Omega0(self):
        """ 
        Set the angular frequency 

        """
        self.Omega0 = 2*np.pi/self.NS_P0

    def Eval_Tc(self):
        """
        Set the critical time Tc
        eq. (5) of Zhang & Meszaros (2001)

        """
        prefac = (self.alpha+self.q+1)
        term2 = prefac*(self.Eimp/(self.L_em0*self.T0))**(1./prefac)
        
        self.Tc = self.T0*max(1,term2)
        self.Tc_units = 's'

    def Eval_T_em(self):
        """
        Set the dipole spin-down time
        eq. (6) of Zhang & Meszaros (2001)

        """
        num = 3*self.lightspeed**3*self.I
        den = self.NS_B**2*self.NS_radius**6*self.Omega0**2

        self.T_em = num/den
        self.T_em_units = 's'
        
    def Eval_L_em0(self):
        """
        Set the plateau luminosity
        eq. (8) of Zhang & Meszaros (2001)

        """
        num = self.I*self.Omega0**2
        den = 2*self.T_em

        self.L_em0 = num/den
        self.L_em0_units = 'ergs/s'

    def LC_radius(self,Omega=None):
        """
        Light cylinder radius (for a given NS rotation)

        """
        if Omega is None:
            Omega=self.Omega0
        out = self.lightspeed/Omega
        return out

    def Magnetospheric_radius(self,T=0,Omega=None):
        """
        Magnetospheric radius 

        """
        if Omega is None:
            Omega=self.Omega0
        Mdot = self.Accretion_rate(T)
        r_lc = self.LC_radius(Omega)
        out = self.mu**(4./7) * (self.gravconst*self.NS_mass)**(-1./7) * Mdot**(-2./7)
        if isinstance(T,np.ndarray):
            mask = out > 0.999*r_lc
            out[mask] = 0.999*r_lc[mask]
        else:
            out = min(out,0.999*r_lc)
        return out

    def Corotation_radius(self,Omega=None):
        """
        Corotation radius (for a given NS mass and spin)

        """
        if Omega is None:
            Omega=self.Omega0
        out = (self.gravconst * self.NS_mass/ Omega**2)**(1./3)
        return out

    def E_rot(self,Omega=None):
        """ 
        Rotational energy of the NS

        """
        if Omega is None:
            Omega=self.Omega0
        out=0.5*self.I*Omega**2
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
        mdot_floor=1e-10
        out = self.Mdot0 * np.exp(-T / self.viscous_time)
        if isinstance(T,np.ndarray):
            out[out<mdot_floor] = mdot_floor
        else:
            out = max(out,mdot_floor)
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
        out[beta>0.27] = 0.

        return out

    def Omega_dot(self,Omega,T):
        """
        Time derivative of the NS spin used in the propeller model

        """
        Mdot = self.Accretion_rate(T)

        r_lc = self.LC_radius(Omega)
        r_mag = self.Magnetospheric_radius(T,Omega)
        r_corot = self.Corotation_radius(Omega)

        Ndip = self.Torque_spindown(T,Omega)
        Nacc = self.Torque_accretion(T,Omega)

        return (Ndip + Nacc)/self.I

    def Eval_Omega(self,T):
        """
        Propeller model from Gompertz et al. 2014

        """
        ## Time integration with LSODA from FORTRAN library ODEPACK
        Omega=odeint(self.Omega_dot,self.Omega0,T)[:,0]
        self.Omega = Omega

    def Eval_critical_angular_velocity(self):
        """
        Sun, Zhang & Gao (2017)
        eq. (25)

        NS collapse for Omega < Omega_c (P>Pc)

        Rem: assume constant NS mass

        """
        num = self.NS_mass - self.EoS_Mtov

        if num<0:
            ## then NS always stable
            self.Omega_c = -1.

        else:
            den = self.EoS_alpha * self.EoS_Mtov
            Pc = (num/den)**(1./self.EoS_beta)
            self.Omega_c = 2*np.pi/Pc
        
    ############################################
    ### Lightcurve definitions
    ### (contribution from dipole,
    ### accretion + propeller, radiative losses)
    ############################################
    def Eval_L_tot(self,T):
        """
        Luminosity function 
        sum of the different contributions
        eq(1)

        """
        Omega = self.Omega
        self.L_tot = self.L_dip + self.L_prop + self.L_rad

        self.L_tot_units = 'erg/s'

    def Eval_L_rad(self,T):
        """
        Loss function
        Limp * T**(-alpha)
        """
        self.L_rad = self.Eimp * (T/self.T0)**(-self.alpha)
        
    def Eval_L_dip(self):
        """
        Dipole spindown luminosity for a generic spin evolution
        """
        #####################################################
        ## Inconsistent prescription in Gompertz
        ## (when modified spindown is used, Bucciantini 2006)
        #####################################################
        #self.L_dip = self.eta_dip * 1./6. * self.mu**2 * self.Omega**4 / self.lightspeed**3

        self.L_dip = -self.NS_eta_dip * self.N_dip * self.Omega

    def Eval_L_prop(self,T):
        """
        Propeller Luminosity
        """
        ## check not necessary ?
        if (self.eta_prop > 0): 
            Omega=self.Omega
            Mdot=self.Accretion_rate(T)
            r_mag=self.r_mag
            Nacc=self.N_acc
            out = self.eta_prop * (- Nacc*Omega - self.gravconst*self.NS_mass*Mdot/r_mag )
            out[out<0.] = 0.
            self.L_prop = out
        else:
            self.L_prop = 0. * T
    
    def L_em(self,T):
        """
        Source function due to dipole radiation
        eq. (7) of Zhang & Meszaros (2001)
        """
        out = self.NS_eta_dip*self.L_em0/(1.+T/self.T_em)**2

        return out
    
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

    def Eval_torques(self,T):
        """
        Compute the various torques
        """
        Omega=self.Omega
        self.N_dip = self.Torque_spindown(T,Omega)
        self.N_acc = self.Torque_accretion(T,Omega)

    def Eval_diag(self,T):
        """
        Compute various derived quantities
        """
        Omega = self.Omega
        self.Mdot = self.Accretion_rate(T)
        self.fast = (self.r_mag / self.r_cor)**1.5
        self.beta = self.E_rot(Omega)/abs(self.E_bind())

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
        fig,ax = plt.subplots()

        ax.loglog(T,self.L_tot,'r-',linewidth=3.0,label=r'$L_{tot}$') 
        ax.loglog(T,self.L_rad,'b--',linewidth=2.0,label=r'$L_{imp}$')
        ax.loglog(T,self.L_dip,'k-.',label=r'$L_{dip}$')
        #ax.loglog(T,self.L_em(T),'y-.',label=r'$L_{em}$')
        if self.eta_prop > 0:
            ax.loglog(T,self.L_prop,'g:',label=r'$L_{prop}$')

        ############
        ### labels
        ############
        ax.legend(fontsize='x-large')
        ax.set_ylabel(r'Luminosity [erg/s]')
        ax.set_xlabel(r'time [s]')

        ##############################
        ### set axis limits by hand...
        ##############################
        #ax.set(xlim=[1.,1e5],ylim=[1e42,1e52])

        plt.tight_layout()
        if savefig:
            ## implement specific filename generator
            ## here if needed
            plt.savefig(filename)

    def PlotRadii(self,T):
        """
        Plot characteristic radii as a function of time
        """
        fig,ax = plt.subplots()
        
        ### Plot of radii (magnetospheric, corotation, light-cylinder)
        ax.loglog(T,self.r_cor,label=r'$r_{corot}$')
        ax.loglog(T,self.r_lc,label=r'$r_{lc}$')
        if self.eta_prop > 0:
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
                   filename='out.fits',
                   outputs = ('time','L_tot'),
                   format='fits',
                   overwrite=True,
                   **kwargs):
        """
        write some outputs as columns in a file 

        Parameter:
        ----------

        filename : string
                like 'path/filename.ext'

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
                    format=format,
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

    #Time array
    #time = np.logspace(0,6,200)

    ## modelling of GRB 061006 with dipole + power law by Gompertz et al 2013
    GRB_061006 = {}
    GRB_061006['T0'] = 200
    GRB_061006['NS_P'] = 24.2e-3
    GRB_061006['NS_B'] = 14.1e15
    GRB_061006['alpha'] = 3.24
    GRB_061006['Mdisk']=0.
    GRB_061006['NS_eta_dip']=1.
    GRB_061006['eta_prop']=0.

    ## modelling of GRB 061006 with both propeller and dipole by Gompertz et al (2014)
    GRB_061006prop = {}
    GRB_061006prop['T0'] = 4e0
    GRB_061006prop['NS_P'] = 1.51e-3
    GRB_061006prop['NS_B'] = 1.48e15
    GRB_061006prop['alpha'] = 5.0
    GRB_061006prop['Mdisk']=2.01e-2
    GRB_061006prop['Rdisk']=400.e5
    GRB_061006prop['NS_eta_dip']=0.05
    GRB_061006prop['eta_prop']=0.4
    #GRB_061006prop['time']=time

    #GRBname = 'GRB061006'
    GRBname = 'GRB061006prop'

    if GRBname == 'GRB061006':
        grb = GRB(**GRB_061006)
        grb.PlotLuminosity(grb.time)
        grb.PlotRadii(grb.time)

    if GRBname == 'GRB061006prop':
        grb = GRB(**GRB_061006prop)
        grb.PlotLuminosity(grb.time)
        ### reset axes by hand
        ax = plt.gca()
        ax.set(xlim=[1.,1e5],ylim=[1e42,1e52])

        grb.PlotRadii(grb.time)

        #grb.WriteTable()

    plt.show()
