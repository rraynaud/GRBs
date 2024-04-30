#!/usr/bin/env python


"""
functions to produce some injections parameters for GW waveforms

"""

import sys
import os
import numpy as np
from astropy.table import QTable
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy_healpix as ah
import matplotlib.pyplot as plt
import json
import itertools

def unifom_radec(nwave):
    """
    produce randomly sky localisation in RA Dec
    :param: nwave: number of sky localisation to produce
    :return: astropy sky coordinates objects
    """

    #flat in RA
    ra = np.random.uniform(low=0, high=2*np.pi, size=nwave)
    #then flat in cos dec
    cosdec = np.random.uniform(low=-1, high=1., size=nwave)
    dec = np.arccos(cosdec)
    mask = dec > np.pi/2
    dec[mask] = dec[mask] - np.pi

    return SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')

def uniform_extragalac(nwave, lat_out=0):
    """
    produce randomly sky localisation in galactic coordinates and 
    transform back to RA-Dec
    we can also exclude a range in latitudes for galactic plane
    :param: nwave: number of sky localisation to produce
    :param: lat_out: latitude range to be excluded ie galactic disk in degrees
    :return: ra (right ascension) and dec (declination) numpy array
    both arrays in degrees
    """

    gal_lon = np.random.uniform(low=0, high=2*np.pi, size=nwave)
    gal_lat_lim = np.cos(np.deg2rad(lat_out))
    cos_gal_lat = np.array([])
    while len(cos_gal_lat)<=nwave: 
        lattest = np.random.uniform(low=-1, high=1., size=nwave)
        mask = np.abs(lattest)<gal_lat_lim
        #print(len(lattest[mask]))
        cos_gal_lat = np.concatenate((cos_gal_lat, lattest[mask]))

    gal_lat = np.arccos(cos_gal_lat[0:nwave])
    mask = gal_lat > np.pi/2
    gal_lat[mask] = gal_lat[mask] - np.pi

    return SkyCoord(l=gal_lon*u.rad, b=gal_lat*u.rad, frame='galactic')

def uniform_cos(max=np.pi):
    """
    function to produce an uniform distribution in cos 
    if max is np.pi/2, do the proper modification
    :return theta: angle in radians
    """

    costhea=ta = np.random.uniform(low=-1, high=1., size=nwave)
    theta = np.arccos(costheta)
    mask = theta > max
    theta[mask] = theta[mask] - np.pi

    return theta

def get_prob_size(skymap, prob_lim=[0.9]):
    """
    taken from EM follow-up guide https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html
    function to compute the prob_lim sky error region
    :param: prob_lim : limite in probability for the sky error region
    :param: skymap : QTable with the probability - used with multiorder fits
    :return: size error region in deg2
    """    

    skymap.sort('PROBDENSITY', reverse=True)
    level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
    pixel_area = ah.nside_to_pixel_area(ah.level_to_nside(level))
    prob = pixel_area * skymap['PROBDENSITY']
    cumprob = np.cumsum(prob)

    areas=[]
    for prob in prob_lim:
        i = cumprob.searchsorted(prob)
        area_p = pixel_area[:i].sum()
        areas.append(area_p.to_value(u.deg**2))

    return areas

def loop_fitsfile(folder="./", probs=[0.9]):
    """
    function to loop on all skymap fits file in a given directory
    by default the current directory
    and scan the error region
    """

    results = []
    # find all fits file
    results += [each for each in os.listdir(folder) if each.endswith('.fits')]

    output=[]

    for filen in results:
        skymap = QTable.read(filen)
        sizemap = get_prob_size(skymap,prob_lim=probs)
        print(filen + " " + str(sizemap))
        tmp=[[filen[:-5]],sizemap[:]]
        merged = list(itertools.chain.from_iterable(tmp))
        output.append(merged)

    return output

def loop_snrfile(folder="./"):
    """
    function to loop on all the snr file and retrieved SNR for all detectors
    """

    results = []
    # find all fits file
    results += [each for each in os.listdir(folder) if each.endswith('snr.json')]

    output = []

    for filen in results:
        with open(filen) as f:
            snr_itf = json.load(f)
            output.append([filen[:-9],snr_itf["H1"],snr_itf["L1"],snr_itf["V1"]])
        f.close()

    return output

def plot_sky_radec(skys):
    """
    test function uniform random in RA-Dec and plot result
    :param: skys: Sky coordinates object

    """

    plt.figure(figsize=(8,4.2))
    plt.subplot(111, projection="aitoff")
    plt.grid(True)
    plt.plot(skys.icrs.ra.wrap_at(180 * u.deg).radian, skys.icrs.dec.radian, 'o', markersize=2, alpha=0.3)
    plt.show()

def plot_sky_galac(skys):
    """
    test function uniform random in galac and plot result
    :param: skys: Sky coordinates object

    """

    plt.figure(figsize=(8,4.2))
    plt.subplot(111, projection="aitoff")
    plt.grid(True)
    plt.plot(skys.galactic.l.wrap_at(180 * u.deg).radian, skys.galactic.b.radian, 'o', markersize=2, alpha=0.3)
    plt.show()

def produce_json(skys):
    """
    produce json entry based on sky coordinates object
    :param: skys: Sky coordinates object 

    """

    data = {
        "ra": skys.icrs.ra.degree.tolist(),
        "dec": skys.icrs.dec.degree.tolist()
    }
     
    print(data)

    with open('radec.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
#"""
#script that will produce uniform distribution in galactic coordinates (and exclude galactic disk)
#and save them into radec.json file
#"""

    if len(sys.argv) > 2:
        skys = uniform_extragalac(int(sys.argv[1]), lat_out=float(sys.argv[2]))
        produce_json(skys)
    else:
        print('missing arguments : number of positions to simulate, exclusion galactic latitude')

    
