#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Transient Lightcurve Modelling

Based on Zhang & Mezaros (2001)
"""
__author__="GRBs: Guilet, Raynaud, Bugli"
__email__ ="jerome.guilet@cea.fr ; raphael.raynaud@cea.fr ; matteo.bugli@cea.fr"
####################################
import os,sys
import numpy as np
import matplotlib as mpl
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
                 eta_dip=1,
                 T0=10,
                 Eimp=1,
                 alpha=0,
                 B=1e15,
                 P=1e-3,
                 M=2.7846e33,# 1.4 Msun
                 R=1e6,
                 Mdisk=1.98855e32, # 0.1 Msun
                 Rdisk=500.0e5, # 500 km
                 alpha_disk=0.1, # disk viscosity parameter
                 cs=1.e7, # sound speed in the disk (100km/s)
                 eta_prop=1,
                 tag='', 
                 verbose=True):
        """
        Parameters
        ----------
        B : float
                magnetar magnetic field
        
        P : float
                magnetar period

        M : float
                magnetar mass    

        R : float
                magnetar radius

        eta_dip : float
                dipole efficiency factor                 
        
        Mdisk : float
                disk mass

        Rdisk : float
                disk radius

        alpha_disk : float
                disk viscosity parameter

        cs : float
                sound speed in the disk

        eta_prop : float
                propeller efficiency factor

        tag : string

        verbose : boolean
                print a summary when instantiating 
                a GRB object
 
        """
        super(GRB, self).__init__()
        ###########################################
        ### input : set inputs and their dimensions
        ###########################################
        self.eta_dip = eta_dip
        self.eta_dip_units = ''
        self.eta_prop = eta_prop
        self.eta_prop_units = ''
        self.Eimp = Eimp
        self.Eimp_units = 'erg'
        self.T0 = T0
        self.T0_units = 's'
        self.alpha = alpha
        self.alpha_units = '' 
        self.B = B
        self.B_units = 'G'
        self.P0 = P
        self.P0_units = 's'
        self.M = M
        self.M_units = 'g'
        self.R = R
        self.R_units = 'cm'
        self.Mdisk0 = Mdisk
        self.Mdisk0_units = 'g'
        self.Rdisk0 = Rdisk
        self.Rdisk0_units = 'cm'
        self.alpha_disk = alpha_disk
        self.alpha_disk_units = ''
        self.cs = cs
        self.cs_units = 'cm/s'
        self.tag = tag
        
        ###########################
        ### astrophysical constants
        ###########################
        self.lightspeed = 299792458e2 # cm/s
        self.Msun = 1.98855e33 # g
        self.gravconst = 6.67259e-8 # cm^3 g^-1 s^-2, gravitational constant
        
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
        self.Eval_viscous_time()
        self.Eval_Mdot0()
        self.Eval_OmegaKep()
        
        ######################
        ### fine tuning
        ######################
        print ('Fine tuning ON...')
        self.Eimp = self.L_em0#*self.T0

        ######################
        ### print a summary
        ######################
        if verbose is True:
            self.Info()

    ##########################################################
    ### DEFINITION OF METHODS
    ##########################################################
    def Info(self):
        """ print a summary"""
        control_param = ('B','P0','R','M','alpha','Eimp','T0','Mdisk0','Rdisk0',
                         'alpha_disk','cs','eta_dip','eta_prop')
        derived_param = ('T_em','Tc','I','L_em0','mu','viscous_time','Mdot0','OmegaKep')
        ### for the layout column width
        lenun = max([len(getattr(self,afield+'_units'))
                     for afield in control_param+derived_param])
        lensy = max([len(afield)
                     for afield in control_param+derived_param])
        
        header = '{:-^%i}'%(lenun+lensy+2+8)
        ligne = '{:%i} {:8.2e} {:%i}'%(lensy,lenun)
        
        print (header.format('Model properties'))
        print (header.format('Input parameters'))
        for afield in control_param:
            info = ligne.format(afield,
                                getattr(self,afield),
                                getattr(self,afield+'_units'))
            print (info)

        print (header.format('Derived quantities'))
        for afield in derived_param:
            info = ligne.format(afield,
                                getattr(self,afield),
                                getattr(self,afield+'_units'))
            print (info)

    ########################################
    ### definition of the derived parameters
    ########################################
    def Eval_MomentOfInertia(self):
        """
        set the magnetar moment of inertia
        """
        #################################
        ### normalisation
        #################################
        ### full sphere formula
        #norm = 2./5
        ### Gompertz (2014)
        norm = 0.35
        #################################
        self.I = norm * self.M*self.R**2
        self.I_units = 'g cm^2'

    def Eval_magnetic_moment(self):
        """
        compute the magnetar magnetic moment
        """
        self.mu = self.B * self.R ** 3
        self.mu_units = "G cm^3"

    def Eval_OmegaKep(self):
        """
        Compute the Keplerian angular frequency at the NS surface
        """
        self.OmegaKep = (self.gravconst * self.M / self.R**3)**0.5
        self.OmegaKep_units = "s^-1"

    def Eval_viscous_time(self):
        """
        compute the viscous timescale of the disk
        """
        self.viscous_time = self.Rdisk0**2 / (3. * self.alpha_disk * self.cs * self.Rdisk0)
        self.viscous_time_units = "s"
        
    def Eval_Mdot0(self):
        """
        Compute the initial mass accretion rate
        """
        self.Mdot0 = self.Mdisk0/self.viscous_time
        self.Mdot0_units = "g/s"
        
    def Eval_Omega0(self):
        """ set the angular frequency """
        self.Omega0 = 2*np.pi/self.P0

    def Eval_Tc(self):
        """
        set the critical time Tc
        eq. (5) of Zhang & Mezraros (2001)
        """
        prefac = (self.alpha+self.q+1)
        term2 = prefac*(self.Eimp/(self.L_em0*self.T0))**(1./prefac)
        
        self.Tc = self.T0*max(1,term2)
        self.Tc_units = 's'

    def Eval_T_em(self):
        """
        set the dipole spin-down time
        eq. 6
        """
        num = 3*self.lightspeed**3*self.I
        den = self.B**2*self.R**6*self.Omega0**2

        self.T_em = num/den
        self.T_em_units = 's'
        
    def Eval_L_em0(self):
        """
        set the plateau luminosity
        eq. 8
        """
        num = self.I*self.Omega0**2
        den = 2*self.T_em

        self.L_em0 = num/den
        self.L_em0_units = 'ergs/s'

        
    ##################################################################################################################
    ### Analytical lightcurve definitions (only for pure dipole, numerical time integration necessary for propeller)
    ##################################################################################################################
    def L_tot(self,T):
        """
        Luminosity function 
        sum of the different contributions
        eq(1)
        """
        return self.L_em(T) + self.L_rad(T)

    def L_rad(self,T):
        """
        Loss function
        Limp * T**(-alpha)
        """
        out = self.Eimp * (T/self.T0)**(-self.alpha)

        return out
        
    def L_em(self,T):
        """
        Source function due to dipole radiation
        eq. 7
        """
        out = self.L_em0/(1.+T/self.T_em)**2

        return out
    
    
    ##################################################################################################################
    ### Numerical time integration (necessary for propeller)                                                                               
    ################################################################################################################## 
    def time_integration(self,time):
        """
        Propeller model from Gompertz et al. 2014
        """
        
        Lrad = self.L_rad(time)
        Ltot = time*0.
        Ldip = time*0.
        Lprop = time*0.
        Nacc = time*0.
        Ndip = time*0.
        Omega = time*0.
        Mdot = time*0.
        r_mag = time*0.
        r_lc = time*0.
        r_corot = time*0.
        fastness = time*0.
        
        i=0
        Omega[0] = self.Omega0
        while (i<len(time)-1):
            # do the time integration
            #Accretion rate on the NS, Eq (13)
            Mdot[i] = max([ 1.0e-10, self.Mdot0 * np.exp(-time[i]/self.viscous_time) ]);
            # radii
            #Light cylinder radius, Eq (5)
            r_lc[i] = self.lightspeed / Omega[i]
            #Alfven radius, Eq (3) + limit to 0.9*r_lc
            r_mag[i] = np.min([0.9*r_lc[i],self.mu**(4./7) * (self.gravconst*self.M)**(-1./7) * Mdot[i]**(-2./7)])
            #Corotation radius, Eq (4)
            r_corot[i] = (self.gravconst * self.M / Omega[i]**2)**(1./3)
            #Eq (6.5)
            fastness[i] = (r_mag[i] / r_corot[i])**1.5
            # Dipole spindown Eq (8) (Eq (2) of Bucciantini et al 2006) torque
            Ndip[i] = - 2./3. * self.mu**2 * Omega[i]**3 / self.lightspeed**3 * (r_lc[i]/r_mag[i])**3
            if (r_mag[i] > self.R):
                Nacc[i] = (1. - fastness[i]) * (self.gravconst * self.M * r_mag[i])**0.5 * Mdot[i]
            else:
                Nacc[i] = (1. - Omega[i]/self.OmegaKep) * (self.gravconst * self.M * r_mag[i])**0.5 * Mdot[i]
            # Luminosities
            #Eq (11), maybe a factor of 2 is missing to account for initial E_kin of the gas (Virial theorem)
            Lprop[i] = self.eta_prop * max([0., - Nacc[i]*Omega[i] - self.gravconst*self.M*Mdot[i]/r_mag[i] ])
            #Eq (14), not consistent with the torque Ndip
            Ldip[i] = self.eta_dip * 1./6. * self.mu**2 * Omega[i]**4 / self.lightspeed**3
            Ltot[i] = Lrad[i] + Lprop[i] + Ldip[i]
            # print "time",time[i],"L_tot",Ltot[i],"L_rad",Lrad[i],"L_dip",Ldip[i],"L_prop",Lprop[i],"Omega",Omega[i],"N_acc",Nacc[i],"Ndip",Ndip[i],"Mdot",Mdot[i],"r_mag",r_mag[i],"r_lc",r_lc[i],"r_corot",r_corot[i],"fastness",fastness[i]
            # time integration of the magnetar rotation frequency
            Omega[i+1] = Omega[i] + (Ndip[i] + Nacc[i])/self.I*(time[i+1]-time[i]) 
            i = i+1
        
        out = {"time":time,"L_tot":Ltot,"L_rad":Lrad,"L_dip":Ldip,"L_prop":Lprop,"Omega":Omega,"N_acc":Nacc[i],"N_dip":Ndip[i],"Mdot":Mdot,"r_mag":r_mag,"r_lc":r_lc,"r_corot":r_corot,"fastness":fastness}

        return out
        
    ########################################
    ### definition of the plotting functions
    ########################################
    
    def PlotLuminosity(self,time):
        """
        plot lightcurve as a function of time

        """
        plt.figure()

        if self.eta_prop == 0.:
            Y = self.L_rad(time)
            plt.loglog(time,Y,ls=':',label=r'$L_{imp}$')
            Y = self.L_em(time)
            plt.loglog(time,Y,ls='--',label=r'$L_{em}$')
            Y = self.L_tot(time)
            plt.loglog(time,Y,label=r'$L_{tot}$')            
        else:
            output = self.time_integration(time)
            plt.loglog(time,output["L_tot"],'r-',linewidth=3.0,label=r'$L_{tot}$')            
            plt.loglog(time,output["L_rad"],'b--',linewidth=2.0,label=r'$L_{imp}$')
            plt.loglog(time,output["L_prop"],'g:',label=r'$L_{prop}$')
            plt.loglog(time,output["L_dip"],'k:',label=r'$L_{em}$')

        ############                                                                                                                                                                              
        ### labels                                                                                                                                                                                
        ############
        plt.legend()
        plt.ylabel(r'Luminosity [erg/s]')
        plt.xlabel(r'time [s]')
        plt.xlim(1.,1e5)
        plt.ylim(1e42,1e52)

        
        if self.eta_prop != 0:
            ### Plot of radii (magnetospheric, corotation, light-cylinder)
            plt.figure(2)
            plt.loglog(output["time"],output["r_mag"],label=r'$r_{mag}$')
            plt.loglog(output["time"],output["r_corot"],label=r'$r_{corot}$')
            plt.loglog(output["time"],output["r_lc"],label=r'$r_{lc}$')
            ### labels
            plt.legend()
            plt.ylabel(r'radius [cm]')
            plt.xlabel(r'time [s]')
            plt.xlim(1.,1e5)
            plt.ylim(1e6,1e9)
            
        ### vlines
        #plt.axvline(self.T_em,label=r'$T_{em}$',ls='--',color='gray')
        #plt.axvline(self.T0,label=r'$T_0$',ls='-',color='gray')
        #plt.axvline(self.Tc,label=r'$T_c$',ls='--',color='r')
        ### hlines
        #plt.axhline(self.L_em0,label=r'$L_{em,0}$',ls='--',color='gray')
        
    
if __name__=='__main__':
    plt.close('all')


    ## modelling of GRB 061006 with dipole + power law by Gompertz et al 2013
    GRB_061006 = {}
    GRB_061006['T0'] = 200
    GRB_061006['P'] = 24.2e-3
    GRB_061006['B'] = 14.1e15
    GRB_061006['alpha'] = 3.24
    GRB_061006['eta_dip']=1.
    GRB_061006['eta_prop']=0.
    
    ## modelling of GRB 061006 with both propeller and dipole by Gompertz et al (2014)
    GRB_061006prop = {}
    GRB_061006prop['T0'] = 4e0
    GRB_061006prop['P'] = 1.51e-3
    GRB_061006prop['B'] = 1.48e15
    GRB_061006prop['alpha'] = 5.0
    GRB_061006prop['Mdisk']=4.0e31  #2.01e-2 Msun 
    GRB_061006prop['Rdisk']=400.e5
    GRB_061006prop['eta_dip']=0.05
    GRB_061006prop['eta_prop']=0.4


    GRBname = 'GRB061006prop'

    if GRBname == 'GRB061006':
        grb = GRB(**GRB_061006)
        time = np.logspace(0,6,200)
        grb.PlotLuminosity(time)
        plt.xlim(0.3,3e5)
        plt.ylim(3e43,2e50)

    if GRBname == 'GRB061006prop':
        grb = GRB(**GRB_061006prop)
        time = np.logspace(0,6,200)
        grb.PlotLuminosity(time)

        
    plt.show()
