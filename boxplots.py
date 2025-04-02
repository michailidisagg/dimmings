"This code computes DEM based on Hannah and Kontar 2012"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as io
import drms
from sys import path as sys_path
from dn2dem_pos import dn2dem_pos
import astropy.time as atime
from astropy import units as u
from aiapy.calibrate import degradation
import warnings
import os
from datetime import datetime
import pandas as pd
#----------------------------------------------------------------------------------------------------------------------
# Initializations
output_folder = f"C:\\Users\\ageli\\Desktop\\paper\\make it to paper\\dem_curve_corrected"
os.makedirs(output_folder, exist_ok=True)

warnings.simplefilter('ignore')
matplotlib.rcParams['font.size'] = 16

channels = [94, 131, 171, 193, 211, 335] * u.angstrom
nc = len(channels)
trin = io.readsav('aia_tresp_en.dat')
gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
dn2ph = gains * np.array([94, 131, 171, 193, 211, 335]) / 3397.
rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18])
num_pix = float(4096**2)

temps = np.logspace(5, 7.5, num=30)
mlogt = [np.mean([np.log10(temps[i]), np.log10(temps[i+1])]) for i in range(len(temps) - 1)]

tresp_logt = np.array(trin['logt'])
nt = len(tresp_logt)
nf = len(trin['tr'][:])
trmatrix = np.zeros((nt, nf))
for i in range(nf):
    trmatrix[:, i] = trin['tr'][i]

client = drms.Client()
keys = ["DATAMEAN", "EXPTIME", "QUALITY", "DATARMS"]
#-----------------------------------------------------------------------------------------------------------------------
# Calculate DEM
def read_dem_input_from_file(filename):
    "Read the input data"
    date, preflare94, preflare131, preflare171, preflare193, preflare211, preflare335, dimming94, dimming131, dimming171, dimming193, dimming211, dimming335, corr_dimming171, corr_dimming193, corr_dimming211 =[],[],[], [], [], [], [], [],[],[],[],[],[],[],[],[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            data = line.strip().split('\t')
            date.append(data[0])
            preflare94.append(float(data[1]))
            preflare131.append(float(data[2]))
            preflare171.append(float(data[3]))
            preflare193.append(float(data[4]))
            preflare211.append(float(data[5]))
            preflare335.append(float(data[6]))

            dimming94.append(float(data[7]))
            dimming131.append(float(data[8]))
            dimming171.append(float(data[9]))
            dimming193.append(float(data[10]))
            dimming211.append(float(data[11]))
            dimming335.append(float(data[12]))

            corr_dimming171.append(float(data[13]))
            corr_dimming193.append(float(data[14]))
            corr_dimming211.append(float(data[15]))
    return date, np.array(preflare94), np.array(preflare131), np.array(preflare171), np.array(preflare193), np.array(preflare211), np.array(preflare335), np.array(dimming94), np.array(dimming131), np.array(dimming171), np.array(dimming193), np.array(dimming211), np.array(dimming335), np.array(corr_dimming171), np.array(corr_dimming193), np.array(corr_dimming211)


def calculate_shotnoise(data, degs):
    shotnoise = np.empty(nc)
    for i in range(nc):
        shotnoise[i] = (dn2ph[i] * data[i] * num_pix) ** 0.5 / dn2ph[i] / num_pix / degs[i]
    return shotnoise

def combine_errors(shotnoise):
    edn_in = np.empty(nc)
    for i in range(nc):
        edn_in[i] = ((rdnse[i] ** 2 + shotnoise[i] ** 2) ** 0.5)/(np.sqrt(200)) #also divide error with the the sqrt num of images
    return edn_in

def get_degradation_factors(timespan):
    time = atime.Time(timespan, scale='utc')
    degs = np.empty(nc)
    for i in range(nc):
        degs[i] = degradation(channels[i], time, calibration_version=10)
    return degs

def write_dem_to_file(filename, date, temps, dem_before, dem_dimming, edem_before, edem_dimming):
    "writes the results in txt file"
    file_exists = os.path.isfile(filename)
    write_header = not file_exists or os.path.getsize(filename) == 0

    with open(filename, 'a') as f:
        if write_header:
            f.write('Date\tTemperature\tDEM_Before\tDEM_Dimming\tEDEM_Before\tEDEM_Dimming\n')
        for i in range(min(len(temps), len(dem_before))):
            f.write(f'{date}\t{temps[i]}\t{dem_before[i]}\t{dem_dimming[i]}\t{edem_before[i]}\t{edem_dimming[i]}\n')


def write_dem_results():
    "calculated DEM"
    date, preflare94, preflare131, preflare171, preflare193, preflare211, preflare335, dimming94, dimming131, dimming171, dimming193, dimming211, dimming335, corr_dimming171, corr_dimming193, corr_dimming211 = read_dem_input_from_file(
        'light_curve_output.txt')

    with open('dem_results_new.txt', 'w'):
        pass  # Clears the file before writing
    with open('dem_results_new_corrected.txt', 'w'):
        pass  # Clears the file before writing

    event_counter = 1
    for i in range(len(date)):
        print(f'Computing event {date[i]}')  # Monitoring

        data0 = [preflare94[i], preflare131[i], preflare171[i], preflare193[i], preflare211[i], preflare335[i]]

        # First computation using original dimmings
        data2 = [dimming94[i], dimming131[i], dimming171[i], dimming193[i], dimming211[i], dimming335[i]]

        # Second computation using corrected dimmings for 171, 193, 211
        data2_corrected = [dimming94[i], dimming131[i], corr_dimming171[i], corr_dimming193[i], corr_dimming211[i],
                           dimming335[i]]
        degs0 = get_degradation_factors(datetime.strptime(f'{date[i]}T12:00:00', '%Y-%m-%dT%H:%M:%S'))
        degs2 = get_degradation_factors(datetime.strptime(f'{date[i]}T12:00:00', '%Y-%m-%dT%H:%M:%S'))

        cor_data0 = data0 / degs0
        cor_data2 = data2 / degs2
        cor_data2_corr = data2_corrected/degs2

        dn_in0 = cor_data0
        dn_in2 = cor_data2
        dn_in2_cor = cor_data2_corr

        shotnoise0 = calculate_shotnoise(data0, degs0)
        shotnoise2 = calculate_shotnoise(data2, degs2)
        shotnoise2_cor = calculate_shotnoise(data2_corrected, degs2)

        edn_in0 = combine_errors(shotnoise0)
        edn_in2 = combine_errors(shotnoise2)
        edn_in2_cor = combine_errors(shotnoise2_cor)

        dem0, edem0, elogt0, _, _ = dn2dem_pos(dn_in0, edn_in0, trmatrix, tresp_logt, temps)
        dem2, edem2, elogt2, _, _ = dn2dem_pos(dn_in2, edn_in2, trmatrix, tresp_logt, temps)
        dem2_cor, edem2_cor, elogt2_cor, _, _ = dn2dem_pos(dn_in2_cor, edn_in2_cor, trmatrix, tresp_logt, temps)

        fig = plt.figure(figsize=(16, 9))
        plt.axvspan(6.0, 6.3, color='gray', alpha=0.3, label='Dimming area')
        plt.errorbar(mlogt, dem0, yerr=edem0, color='r', label='Preflare', fmt='.')
        plt.errorbar(mlogt, dem2, yerr=edem2, color='g', label='Dimming', fmt='.')
        if event_counter not in [4,6,8,9,10]: #select manualy the events with correction
           plt.errorbar(mlogt, dem2_cor, yerr=edem2_cor, color='b', label='Dimming corrected', fmt='.')
        plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
        plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')

        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.title(f"{date[i]}")
        #plt.show()
        frame_filename = os.path.join(output_folder, f'DEM_{date[i]}.png')
        plt.savefig(frame_filename)
        plt.close()

        write_dem_to_file('dem_results_new.txt', date[i], mlogt, dem0, dem2, edem0, edem2)
        write_dem_to_file('dem_results_new_corrected.txt', date[i], mlogt, dem0, dem2_cor, edem0, edem2_cor)

        event_counter = event_counter+1

write_dem_results() #call the function that calculates DEM
#----------------------------------------------------------------------------------------------------------------------
# Make the boxplots
def read_dem_from_file(filename):
    "reads DEM results"
    temps, dem_before, dem_dimming, edem_before, edem_dimming = [], [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header line
            data = line.strip().split('\t')
            temps.append(float(data[1]))
            dem_before.append(float(data[2]))
            dem_dimming.append(float(data[3]))
            edem_before.append(float(data[4]))
            edem_dimming.append(float(data[5]))
    return np.array(temps), np.array(dem_before), np.array(dem_dimming), np.array(edem_before), np.array(edem_dimming)



input_file = 'dem_results_new.txt'
# Read the DEM results from the file
temps, dem_before, dem_dimming, edem_before, edem_dimming = read_dem_from_file(input_file)
# Calculate the DEM ratios (Before/Dimming)
dem_ratios = dem_dimming / dem_before


num_temps = len(mlogt)
num_events = len(dem_ratios) // num_temps
# Reshape dem_ratio so each row is an event, and each column corresponds to a temperature
dem_ratio_reshaped = np.reshape(dem_ratios, (num_events, num_temps))
print(dem_ratio_reshaped.shape)
print(dem_ratio_reshaped)
# Create a boxplot where each temperature has its own set of values
plt.figure(figsize=(16, 9))
plt.subplot(121)
plt.boxplot([dem_ratio_reshaped[:, i] for i in range(num_temps)],
            labels=[f'{mlogt[i]:.2f}' for i in range(num_temps)], showfliers=False)
plt.xlim(12.5,15.5)
plt.ylim(0.75,1.20)
yticks = np.arange(0.75, 1.21, 0.05)
plt.yticks(yticks)
plt.axhline(1.0, color='black', linestyle='--')
plt.grid()
plt.xlabel('Temperature ($\log_{10}T$ [K])')
plt.ylabel('DEM Ratio (Dimming/Preflare)')
plt.title('Uncorrected Input Lightcurves')
plt.xticks(rotation=45)
plt.tight_layout()

input_file = 'dem_results_new_corrected.txt'
# Read the DEM results from the file
temps, dem_before, dem_dimming, edem_before, edem_dimming = read_dem_from_file(input_file)
# Calculate the DEM ratios (Before/Dimming)
dem_ratios = dem_dimming / dem_before
# Create a dictionary to store DEM ratios for each temperature

# Assume mlogt and dem_ratio are already loaded
num_temps = len(mlogt)  # Should be 30 or 29
num_events = len(dem_ratios) // num_temps  # Total events
# Reshape dem_ratio so each row is an event, and each column corresponds to a temperature
dem_ratio_reshaped = np.reshape(dem_ratios, (num_events, num_temps))
print(dem_ratio_reshaped.shape)
print(dem_ratio_reshaped)
# Create a boxplot where each temperature has its own set of values
plt.subplot(122)
plt.boxplot([dem_ratio_reshaped[:, i] for i in range(num_temps)],
            labels=[f'{mlogt[i]:.2f}' for i in range(num_temps)], showfliers=False)
plt.xlim(12.5,15.5)
plt.ylim(0.75,1.20)
yticks = np.arange(0.75, 1.21, 0.05)
plt.yticks(yticks)
plt.axhline(1.0, color='black', linestyle='--')
plt.grid()
plt.xlabel('Temperature ($\log_{10}T$ [K])')
plt.ylabel('DEM Ratio (Dimming/Preflare)')
plt.title('Corrected Input Lightcurves')
plt.xticks(rotation=45)
plt.tight_layout()

plt.suptitle('DEM(T) Ratio After Solar Flare (Dimming) over Preflare')
#plt.show()

num_temps = dem_ratio_reshaped.shape[1]

# Iterate over each temperature bin to find outliers
for i in range(num_temps):
    data = dem_ratio_reshaped[:, i]  # Extract data for temperature bin i
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1  # Compute IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Print outliers for this temperature bin
    print(f"Temperature log10(T) = {mlogt[i]:.2f} K:")
    if len(outliers) > 0:
        print(f"  Outliers: {outliers.tolist()}\n")
    else:
        print("  No outliers.\n")
#-----------------------------------------------------------------------------------------------------------------------
#intensity boxplots
data_file = "light_curve_output.txt"  #intensity input file
df = pd.read_csv(data_file, sep="\t")

# Compute the dimming/pre-event ratios
dimming_ratio_original = df.iloc[:, 7:13].div(df.iloc[:, 1:7].values)
dimming_ratio_corrected = dimming_ratio_original.copy()
dimming_ratio_corrected.iloc[:, 2:5] = df.iloc[:, 13:16].div(df.iloc[:, 3:6].values)

# Create the boxplots
plt.figure(figsize=(12, 6))

# First boxplot (original dimming values)
plt.subplot(1, 2, 1)
plt.boxplot(dimming_ratio_original.values, labels=["94", "131", "171", "193", "211", "335"], showfliers=False)
plt.title("Uncorrected")
plt.ylabel("Intensity Ratio (Dimming/Preflare)")
plt.xlim(2.5,5.5)
plt.ylim(0.91,1.02)
plt.axhline(1.0, color='black', linestyle='--')
plt.xlabel('Wavelength ($\AA$)')
plt.grid(True)

# Second boxplot (corrected values)
plt.subplot(1, 2, 2)
plt.boxplot(dimming_ratio_corrected.values, labels=["94", "131", "171", "193", "211", "335"], showfliers=False)
plt.title("Corrected")
plt.ylabel("Intensity Ratio (Dimming/Preflare)")
plt.xlim(2.5,5.5)
plt.ylim(0.91,1.02)
plt.axhline(1.0, color='black', linestyle='--')
plt.grid(True)
plt.xlabel('Wavelength ($\AA$)')
plt.suptitle('Intensity Ratio After Solar Flare (Dimming) over Preflare')
plt.tight_layout()
plt.show()
