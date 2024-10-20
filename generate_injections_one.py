#!/usr/bin/env python

"""
Script to produce GW waveforms and associated skymaps
it is based on pycbc scripts 
Several parameters needs to be defined inside a json config file (presently name hardcoded)
at minima the parameters needs to be 'mass1', 'mass2', 'distance'
the optional parameters, if not defined, will be randomly produced
'spin1z', spin of first component
'spin2z', spin of second component
'ra', right ascension
'dec', declination
'tc', time of coalescence 

"""

import sys,os
import numpy as np
import json
from datetime import datetime
from pycbc.io import FieldArray
from pycbc.inject import InjectionSet

dtype = [('mass1', float), ('mass2', float),
         ('spin1z', float), ('spin2z', float),
         ('tc', float), ('distance', float),
         ('ra', float), ('dec', float),
         ('coa_phase', float),
         ('polarization', float),
         ('inclination', float),
         ('approximant', 'S32')]

static_params = {'f_lower': 17.,
                 'f_ref': 17.,
                 'taper': 'startend'
                }
#                 'inclination': 0.,
#                 'coa_phase': 0.,
#                 'polarization': 0.}

with open('./config.json') as filename:
    config = json.load(filename)

#number os simulations
nwave = len(config['mass1'])

samples = FieldArray(nwave, dtype=dtype)

# masses and spins are intended to match the highest
# and lowest mass templates in the template bank
# Last injection is designed to be found as an EM-bright single
samples['mass1'] = config['mass1']
samples['mass2'] = config['mass2']

#inclination is defined between 0 and pi
#0 means face-on, jet oriented to our direction
#pi means face-away 
costheta = np.random.uniform(low=-1, high=1., size=nwave)
theta = np.arccos(costheta)
#mask = theta > np.pi/2
#theta[mask] = theta[mask] - np.pi
samples['inclination'] = theta

samples['coa_phase'] = np.random.uniform(low=0, high=2*np.pi, size=nwave) 
samples['polarization'] = np.random.uniform(low=0, high=2*np.pi, size=nwave)

#check if spin keyword exists, otherwise randmly choose it
try:
    samples['spin1z'] = config['spin1z']
except KeyError as err:
    print('spin1z not defined, we will randomly choose them')
    config['spin1z'] = np.random.uniform(low=-0.05, high=0.05, size=nwave)
    samples['spin1z'] = config['spin1z']

try:
    samples['spin2z'] = config['spin2z']
except KeyError as err:
    print('spin2z not defined, we will randomly choose them')
    config['spin2z'] = np.random.uniform(low=-0.05, high=0.05, size=nwave)
    samples['spin2z'] = config['spin2z']

# check time of coalescence
# presently use random date (in 2020) and spread on one week
# round the precision to 2 digits after unity
try:
    samples['tc'] = config['tc']
except KeyError as err:
    print('tc not defined, we will randomly choose them')
    tcini = 1272790260.0
    config['tc'] = tcini + 8640000. * np.round(np.random.uniform(low=.1, high=.9, size=nwave),5)
    samples['tc'] = config['tc']

#np.set_printoptions(precision=15)
#print(config['tc'])

samples['distance'] = config['distance']

try:
    samples['ra'] = np.deg2rad(config['ra'])
    samples['dec'] = np.deg2rad(config['dec'])
except KeyError as err:
    print('RA-Dec positions not defined, we will randomly choose them')
    cmd = "./utils_gw.py " + str(nwave) 
    os.system(cmd)
    with open('./radec.json') as filenameb:
        pos = json.load(filenameb)

    config['ra'] = pos['ra']
    config['dec'] = pos['dec']
    samples['ra'] = np.deg2rad(config['ra'])
    samples['dec'] = np.deg2rad(config['dec'])

# need to understand when using which type
# presently use only SpinTaylorT4
try:
    samples['approximant'] = config['approximant']
except KeyError as err:
    print('no approximant defined, use only SpinTaylorT4')
    config['approximant'] = ["SpinTaylorT4"] * nwave
    samples['approximant'] = config['approximant']

InjectionSet.write('injections.hdf', samples, static_args=static_params,
                   injtype='cbc', cmd=" ".join(sys.argv))

gps = datetime(1980,1,6)
time_n = str(round((datetime.now()-gps).total_seconds()))

cmd_line = 'cp injections.hdf injections' + '_' + str(nwave) + '_' + time_n + '.hdf'
os.system(cmd_line)

f = open('output_' + str(nwave) + '_' + time_n + '.txt', "w")

for i in range(0,len(config['tc'])):
    gps = str(round(config['tc'][i]))
    print(gps)

    m1 = str(round(samples['mass1'][i],2))
    m2 = str(round(samples['mass2'][i],2))
    sp1z = str(round(samples['spin1z'][i],3))
    sp2z = str(round(samples['spin2z'][i],3))

    incli = str(round(samples['inclination'][i],3))

    dist = str(round(samples['distance'][i]))
    pos = str(round(samples['ra'][i],3)) + ' ' + str(round(samples['dec'][i],3))

    cmd_line = m1 + ' ' + m2 + ' ' + sp1z + ' ' + sp2z + ' ' + gps + ' ' + incli + ' ' + dist + ' ' + pos + '\n'
    f.write(cmd_line)

    cmd_line = './pycbc_make_skymap.py --trig-time ' + gps + \
               ' --fake-strain H1:aLIGOaLIGO175MpcT1800545 ' + \
               'L1:aLIGOaLIGO175MpcT1800545 ' + \
               'V1:AdVO3LowT1800545 ' + \
               '--injection-file injections.hdf --thresh-SNR 5.5 ' + \
               '--f-low 20 --mass1 ' + str(config['mass1'][i]) + \
               ' --fake-strain-seed H1:1234 L1:2345 V1:3456 ' + \
               ' --mass2 ' + str(config['mass2'][i]) + \
               ' --spin1z ' + str(config['spin1z'][i]) + \
               ' --spin2z ' + str(config['spin2z'][i]) + ' --ifos H1 L1 V1 ' + \
               '--ligolw-event-output coinc_simulated_data.xml'

    print(cmd_line)
    os.system(cmd_line)

    filepng = gps + '_skymap_zoom.png'
    fitsfile = gps + '.fits'

    cmd_line = 'ligo-skymap-plot --output ' + filepng + \
               ' --projection zoom --projection-center ' + \
               '\''+str(config['ra'][i]) + 'deg ' + str(config['dec'][i]) + 'deg\' ' + \
               '--zoom-radius 20deg --contour 50 90 --annotate ' + \
               '--radec ' + str(config['ra'][i]) + ' ' + str(config['dec'][i]) + \
               ' ' + fitsfile

    print(cmd_line)
    os.system(cmd_line)

f.close()

