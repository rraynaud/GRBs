Python scripts to generate sky maps for a HLV network

requirements
install pycbc, using pip install pycbc

code used :
utils_gw.py: functions needed for preparing injections, it contains random
sky generation, either in RA-Dec or in galactic coordinates and associated
plot functions 
random generation in galactic coordinates allows to exclude low latitudes
to avoid galactic plane

generate_injections_one.py : script that will generate waveforms (restricted to BNS), 
compute SNR, and produce associated sky maps by using bayestar
it uses a config.json file to include the different parameters, the minimum
infos needed are :
mass1, mass2 and distance, for example
{
 "mass1": [1.1331687,1.45],
 "mass2": [1.010624,1.3],
 "distance": [79,85]
}

others parameters can be provided, if not they will be randomly provided

a full example of config is
{
 "mass1": [1.1331687],
 "mass2": [1.010624],
 "spin1z": [0.029544285],
 "spin2z": [0.020993788],
 "tc": [1272790260.1],
 "distance": [79],
 "ra": [45],
 "dec": [-10],
 "approximant": ["SpinTaylorT4"]
} 

