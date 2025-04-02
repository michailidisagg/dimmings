import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
from scipy.stats import t
def read_dem_from_file(filename):
    dates, temps, dem_before, dem_dimming, edem_before, edem_dimming = [], [], [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header line
            data = line.strip().split('\t')
            dates.append(data[0])
            temps.append(float(data[1]))
            dem_before.append(float(data[2]))
            dem_dimming.append(float(data[3]))
            edem_before.append(float(data[4]))
            edem_dimming.append(float(data[5]))
    return np.array(dates), np.array(temps), np.array(dem_before), np.array(dem_dimming), np.array(edem_before), np.array(edem_dimming)

def read_masses_from_file(filename):
    masses = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            data = line.strip().split(' ')
            masses.append(float(data[1]))  # Assuming the masses are in the second column
    return np.array(masses)

def integral(x, y):
    integral = 0
    log_x_min = 5.7
    log_x_max = 6.3
    mask = (x >= log_x_min) & (x <= log_x_max)

    if np.any(mask):
        x_normal = 10 ** x[mask]
        for i in range(len(x_normal) - 1):
            integral += (y[mask][i] + y[mask][i + 1]) * (x_normal[i + 1] - x_normal[i]) / 2

    return integral

def integral_2(x, dem_before, dem_dimming):
    integral = 0
    log_x_min = 6.0
    log_x_max = 6.3
    mask = (x >= log_x_min) & (x <= log_x_max)
    if np.any(mask):
        x_normal = 10 ** x[mask]
        y_diff = dem_before[mask] - dem_dimming[mask]  # Calculate the difference
        for i in range(len(x_normal) - 1):
            integral += (y_diff[i] + y_diff[i + 1]) * (x_normal[i + 1] - x_normal[i]) / 2

    return integral

def integral_err(x, edem_before, edem_dimming):
    integral = 0
    log_x_min = 6.0
    log_x_max = 6.3
    mask = (x >= log_x_min) & (x <= log_x_max)


    if np.any(mask):
        x_normal = 10 ** x[mask]
        y_diff = edem_before[mask]**2 + edem_dimming[mask]**2  # Calculate the difference
        for i in range(len(x_normal) - 1):
            integral += (y_diff[i] + y_diff[i + 1]) * (x_normal[i + 1] - x_normal[i]) / 2

    return np.sqrt(integral)

def process_events(dates, temps, dem_before, dem_dimming, edem_before, edem_dimming):
    events_results = []
    em_before = []
    em_dimming = []
    em_diff = []
    em_diff_err =[]

    event_pointer = np.arange(0, 30*10, 30)
    for i in event_pointer:
        integral_before = integral(temps[i:30 + i], dem_before[i:30+i])
        integral_dimming = integral(temps[i:30 + i], dem_dimming[i:30 + i])
        integral_diff = integral_2(temps[i:30 + i], dem_before[i:30+i], dem_dimming[i:30 + i])
        integral_error = integral_err(temps[i:30 + i], edem_before[i:30+i], edem_dimming[i:30 + i])

        em_before.append(integral_before)
        em_dimming.append(integral_dimming)
        em_diff.append(integral_diff)
        em_diff_err.append(integral_error)

    em_before = np.array(em_before)
    em_dimming = np.array(em_dimming)
    em_diff = np.array(em_diff)
    em_diff_err = np.array(em_diff_err)

    return em_before, em_dimming, em_diff, events_results, em_diff_err


def save_results_to_file(filename, events_results):
    with open(filename, 'w') as f:
        f.write("Date\tEM_difference\n")
        for date, em_diff in events_results:
            f.write(f"{date}\t{em_diff}\n")

def linear_func(parameters, x):
    a, b = parameters
    return a * x + b
linear_model = Model(linear_func)
def fit_and_plot(cme_mass, em_diff, em_errors):
    # Perform filtering
    mask = cme_mass != 1.4e16 #This is the double events that we exclude
    em_diff = em_diff[mask]
    cme_mass = cme_mass[mask]
    em_errors = em_errors[mask]

    # Apply square root transformation
    # calculate errorbars and input mass for the fit

    errorbars = (2 * cme_mass).copy()
    cme_mass_fit = (2 * cme_mass).copy()
    err_fit_y = cme_mass.copy()

    # the non-halo event (i select it manually because i have few events and i know which one it is)
    errorbars[4] = cme_mass[4]
    cme_mass_fit[4] = 1.5 * cme_mass[4]
    err_fit_y[4] = 0.5 * cme_mass[4]



    data = RealData(em_diff, cme_mass_fit, sx=em_errors, sy=err_fit_y) #Uncomment for fit with errors
    #data = RealData(em_diff, cme_mass_fit) #Uncomment for fit without errors
    odr = ODR(data, linear_model, beta0=[0., 1.])
    out = odr.run()
    slope, intercept = out.beta
    fit_line = linear_func(out.beta, em_diff)

    # Compute R-squared
    residuals = cme_mass_fit - fit_line
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((cme_mass_fit - np.mean(cme_mass_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute standard error of the slope
    se_slope = out.sd_beta[0]  # Standard error of the slope
    slope = out.beta[0]  # Estimated slope

    # Compute t-statistic
    t_stat = slope / se_slope

    # Degrees of freedom
    n = len(em_diff)  # Number of data points
    p = len(out.beta)  # Number of parameters (slope and intercept)
    df = n - p  # Degrees of freedom

    # Compute two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    # Plot
    plt.figure(figsize=(16, 9))
    plt.title('CME Mass as Function of $EM_{difference}$',
              fontsize=20)

    # Scatter plot with error bars
    plt.errorbar(em_diff, cme_mass_fit, xerr=em_errors, yerr=err_fit_y, linestyle=' ', marker=' ', color='red',
                 label='Errorbars')
    plt.scatter(em_diff, cme_mass, color='blue', label='Data', marker='.')

    # Plot fitted line
    plt.plot(em_diff, fit_line, color='green', linestyle='--',
             label=(
                 f'Fit: $y = {intercept / 10 ** int(np.floor(np.log10(abs(intercept)))):.3f} \\times 10^{{{int(np.floor(np.log10(abs(intercept))))}}}+'
                 f'{slope / 10 ** int(np.floor(np.log10(abs(slope)))):.3f} \\times 10^{{{int(np.floor(np.log10(abs(slope))))}}} x$\n'
                 f'$R^2 = {r_squared:.3f}$, $p = {p_value:.3g}$'))

    # Labels and styling
    plt.ylabel('CME Mass ($g$)', fontsize=16)
    plt.xlabel('$EM_{difference}$ ($cm^{-5}$)', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=20, loc='upper left')
    plt.tight_layout()
    plt.show()


# Κύριος κώδικας
data_file = 'dem_results_new_corrected.txt'
masses_file = 'event_masses.txt'
output_file = 'em_difference_results.txt'

dates, temps, dem_before, dem_dimming, edem_before, edem_dimming = read_dem_from_file(data_file)
masses = read_masses_from_file(masses_file)

em_before, em_dimming, em_diff, events_results, em_errors = process_events(dates, temps, dem_before, dem_dimming, edem_before, edem_dimming)
sorted_indices = np.argsort(em_diff)
em_diff_sorted = em_diff[sorted_indices]
masses_sorted = masses[sorted_indices]
fit_and_plot(masses_sorted, em_diff_sorted, em_errors)