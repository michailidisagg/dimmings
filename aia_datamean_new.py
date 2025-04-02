"This code queries SDO/AIA data and creates lightcurves using the DATAMEAN keyword. The goal is to detect dimming in the light curves after a CME event. The code also applies a gradual phase correction."
# Import modules
import drms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.ndimage import gaussian_filter1d
import os
#-----------------------------------------------------------------------------------------------------------------------
# Define output folders
output_folder = f"C:\\Users\\ageli\\Desktop\\paper\\make it to paper\\light_curve_corrected\\Light_curves_corrected_cleaned"
output_txt_folder = 'light_curves_txt_individual'
os.makedirs(output_txt_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
output_file =  'light_curve_output.txt'
#-----------------------------------------------------------------------------------------------------------------------
#for smoothing
def smoothin(light_curve):
    'This function applies a gayssian smoothing to the lightcurves'
    return gaussian_filter1d(light_curve, 5)
    #return light_curve

# Align and trim function
def align_and_trim(reference, target, original_reference, original_target):
    "This function aligns the peaks of the dimming and non-dimming curves and trims the edges"
    ref_peak_index = np.argmax(original_reference)
    target_peak_index = np.argmax(original_target)

    # Compute the shift needed to align the peaks
    shift = ref_peak_index - target_peak_index
    print(f"Reference Peak: {ref_peak_index}, Target Peak: {target_peak_index}, Shift: {shift}")

    # Determine the valid start and end indices for trimming
    max_start = max(0, shift)
    max_end = min(len(reference), len(target) + shift)

    min_start = max(0, -shift)
    min_end = min(len(reference) - shift, len(target))

    # Extract aligned slices ensuring equal length
    aligned_ref = reference[max_start:max_end]
    aligned_target = target[min_start:min_end]

    return aligned_ref, aligned_target

# Read DEM_dates.txt
events = pd.read_csv('dates_cleaned.txt', delim_whitespace=True, header=None, names=['start', 'mid', 'end'])

# Querying keys
client = drms.Client()
keys = ["T_OBS", "DATAMEAN", "EXPTIME", "QUALITY", "PERCENTD"]

# AIA Channels
channels = [
    {"wavelength": 94, "cadence": "12s", "color": 'r', "name": "94A channel"},
    {"wavelength": 131, "cadence": "12s", "color": 'b', "name": "131A channel"},
    {"wavelength": 171, "cadence": "12s", "color": 'g', "name": "171A channel"},
    {"wavelength": 193, "cadence": "12s", "color": 'r', "name": "193A channel"},
    {"wavelength": 211, "cadence": "12s", "color": 'b', "name": "211A channel"},
    {"wavelength": 335, "cadence": "12s", "color": 'g', "name": "335A channel"}
]



with open(output_file, 'w') as f:
    pass #clear the output file

event_counter = 0
# Loop through each event
for idx, event in events.iterrows():

    event_counter = event_counter+1
    start_time = pd.to_datetime(event['start'])
    mid_time = pd.to_datetime(event['mid'])
    end_time = pd.to_datetime(event['end'])

    date = mid_time.strftime('%Y-%m-%d')
    timespan = f"{start_time}/8h@12s"
    print(f'Computing event {date}')

    # Initialize arrays
    fix_335, fix_171, fix_193, fix_211 = [], [], [], []
    prefix_171, prefix_193, prefix_211 = [], [], []

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    plt.suptitle(f"Event at {date}", y=0.98)

    # Loop through channels
    for i, channel in enumerate(channels):
        wavelength, cadence, color, name = channel.values()
        qstr = f"aia.lev1_euv_{cadence}[{timespan}][{wavelength}]"

        # Query data
        data = client.query(qstr, key=keys)
        data = data[(data.QUALITY == 0) & (data.DATAMEAN > 0) & (data.EXPTIME > 1.5)]
        data.index = drms.to_datetime(data.T_OBS)
        norm_data = data["DATAMEAN"] / data["EXPTIME"]


        # Store light curves
        if wavelength == 94:
            before_smooth94 = np.array(norm_data)
            original94 = smoothin(np.array(norm_data))
            if event_counter == 3:
                start94 = np.mean(original94[:25])
                fix_335 = -(start94 - original94) / start94

        if wavelength == 131:
            before_smooth131 = np.array(norm_data)
            original131 = smoothin(np.array(norm_data))

        if wavelength == 335:
            before_smooth335 = np.array(norm_data)
            original335 = smoothin(np.array(norm_data))
            start335 = np.mean(original335[:25])
            if event_counter != 3:
               fix_335 = -(start335 - original335) / start335
        elif wavelength == 171:
            before_smooth171 = np.array(norm_data)
            original171 = smoothin(np.array(norm_data))
            start171 = np.mean(original171[:25])
            prefix_171 = -(start171 - original171) / start171
        elif wavelength == 193:
            before_smooth193 = np.array(norm_data)
            original193 = smoothin(np.array(norm_data))
            start193 = np.mean(original193[:25])
            prefix_193 = -(start193 - original193) / start193
        elif wavelength == 211:
            before_smooth211 = np.array(norm_data)
            original211 = smoothin(np.array(norm_data))
            start211 = np.mean(original211[:25])
            prefix_211 = -(start211 - original211) / start211
        # Plot original data
        ax = axes[i // 3, i % 3]
        norm_data.plot(ax=ax, color=color)
        ax.set_title(name)
        ax.grid()

        # Highlight key event times
        ax.axvspan(start_time, start_time + timedelta(minutes=20), color='red', alpha=0.3)
        ax.axvspan(mid_time - timedelta(minutes=10), mid_time + timedelta(minutes=10), color='blue', alpha=0.3)
        ax.axvspan(end_time - timedelta(minutes=10), end_time + timedelta(minutes=10), color='green', alpha=0.3)

    # Compute scale factors (ensure no division by zero)
    if np.max(fix_335) > 0:
        scale_factor171 = min(np.max(prefix_171) / np.max(fix_335), np.max(fix_335) / np.max(prefix_171))
        scale_factor193 = min(np.max(prefix_193) / np.max(fix_335), np.max(fix_335) / np.max(prefix_193))
        scale_factor211 = min(np.max(prefix_211) / np.max(fix_335), np.max(fix_335) / np.max(prefix_211))


    # Normalize fix_335
    normfix_171 = fix_335 * scale_factor171
    normfix_193 = fix_335 * scale_factor193
    normfix_211 = fix_335 * scale_factor211

    # Align the data
    prefix_171, normfix_171 = align_and_trim(prefix_171, normfix_171, original171, original335)
    prefix_193, normfix_193 = align_and_trim(prefix_193, normfix_193, original193, original335)
    prefix_211, normfix_211 = align_and_trim(prefix_211, normfix_211, original211, original335)

    # Compute fixed light curves
    fix_171 = prefix_171 - normfix_171
    fix_193 = prefix_193 - normfix_193
    fix_211 = prefix_211 - normfix_211

    # Restore original scale
    fix_171 = start171 * (1 + fix_171)
    fix_193 = start193 * (1 + fix_193)
    fix_211 = start211 * (1 + fix_211)
    fix_335 = start335 * (1 + fix_335)

    # Ensure they all have the same size
    min_length = min(len(fix_171), len(fix_193), len(fix_211), len(fix_335))
    fix_171, fix_193, fix_211, fix_335 = fix_171[:min_length], fix_193[:min_length], fix_211[:min_length], fix_335[:min_length]


    # Plot corrected light curves
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.ravel()  # Flatten the 2D array into a 1D array for iteration

    # Define data
    corrected_data_list = [   original94, original131,
        fix_171,
        fix_193,
        fix_211,  original335
    ]
    # Here we chose manually the events that have the correction, the pre-event window and the dimming window
    if event_counter in [3]:
        dimming_start = 650
        dimming_end = 850
    elif event_counter in[11,12]:
        dimming_start = np.argmin(fix_171) - 50
        dimming_end = np.argmin(fix_171) + 150
    elif event_counter in [6,8,9,10]:
        dimming_start = np.argmin(original211) - 100
        dimming_end = np.argmin(original211) + 100
    elif event_counter == 4:
        dimming_start = np.argmin(original193) - 50
        dimming_end = np.argmin(original193) + 150
    else:
        dimming_start = np.argmin(fix_211) - 50
        dimming_end = np.argmin(fix_211) + 150
    print(f'dimming area {dimming_start, dimming_end}')

    colors = ['g', 'b', 'g', 'b', 'g', 'b']  # Extended color list
    titles = ['94A', '131A', '171A', '193A', '211A', '335A']
    original_data_list = [original94, original131, original171, original193, original211, original335]
    before_smooth = [before_smooth94, before_smooth131, before_smooth171, before_smooth193, before_smooth211, before_smooth335,]

    # Iterate through axes and data
    for ax, corrected_data, color, title, original, before_smooth in zip(axes, corrected_data_list, colors, titles,
                                                          original_data_list, before_smooth):
        if event_counter not in [4,6,8,9,10]:
            if title == '171A' or title == '193A' or title == '211A':
                ax.plot(corrected_data, color='red', label='corrected')
        ax.plot(before_smooth[:len(corrected_data)], color='#1f77b4', label='original - no smooth', alpha=1)
        ax.plot(original[:len(corrected_data)], color='#ff7f0e', label='original smooth', alpha=1, linestyle='--')
        ax.axhline(np.mean(original[:25]), linestyle='--', color='black')
        ax.set_title(title)
        ax.set_ylabel('Datamean (Dn/s)')
        ax.set_xlabel('Image index')
        ax.grid()

        # Highlight event periods
        ax.axvspan(0, 200, color='red', alpha=0.3, label='pre-event')
        ax.axvspan(dimming_start, dimming_end, color='green', alpha=0.3, label='dimming')
        ax.legend()

    #extract the preflare and dimming values
    preflare94 = np.mean(original94[:200])
    preflare131 = np.mean(original131[:200])
    preflare171 = np.mean(fix_171[:200])
    preflare193 = np.mean(fix_193[:200])
    preflare211 = np.mean(fix_211[:200])
    preflare335 = np.mean(original335[:200])


    dimming94 = np.mean(original94[dimming_start:dimming_end])
    dimming131 = np.mean(original131[dimming_start:dimming_end])
    dimming171 = np.mean(original171[dimming_start:dimming_end])
    dimming193 = np.mean(original193[dimming_start:dimming_end])
    dimming211 = np.mean(original211[dimming_start:dimming_end])
    dimming335 = np.mean(original335[dimming_start:dimming_end])
    if event_counter not in [4,6,8,9,10]:
        corr_dimming171 = np.mean(fix_171[dimming_start:dimming_end])
        corr_dimming193 = np.mean(fix_193[dimming_start:dimming_end])
        corr_dimming211 = np.mean(fix_211[dimming_start:dimming_end])
    else:
        corr_dimming171 = dimming171
        corr_dimming193 = dimming193
        corr_dimming211 = dimming211


    with open(output_file, 'a') as f:
        f.write(f'{date}\t{preflare94}\t{preflare131}\t{preflare171}\t{preflare193}\t{preflare211}\t{preflare335}\t{dimming94}\t{dimming131}\t{dimming171}\t{dimming193}\t{dimming211}\t{dimming335}\t{corr_dimming171}\t{corr_dimming193}\t{corr_dimming211}\n')




    plt.suptitle(f'Start Time {start_time}')
    plt.tight_layout()
    frame_filename = os.path.join(output_folder, f'Light_curve_corrected_{date}.png')
    plt.savefig(frame_filename)
    plt.close()
    plt.show()