import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.odr import *
#-----------------------------------------------------------------------------------------------------------------------
#read DEMs and their errors
def read_dem_from_file2(filename):
    dem_before, dem_dimming, err_before, err_dimming = [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            data = line.strip().split('\t')
            dem_before.append(float(data[2]))
            dem_dimming.append(float(data[3]))
            err_before.append(float(data[4]))
            err_dimming.append(float(data[5]))
    return np.array(dem_before), np.array(dem_dimming), np.array(err_before), np.array(err_dimming)
#-----------------------------------------------------------------------------------------------------------------------
#read event's masses
def read_mass_file(filename):
    cme_mass = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            data = line.strip().split(' ')
            cme_mass.append(float(data[1]))

    return np.array(cme_mass)
#-----------------------------------------------------------------------------------------------------------------------
def dem_ratio_error(dem_before, dem_dimming, err_before, err_dimming):
    return 0.5 * np.sqrt((err_dimming**2 / (dem_before * dem_dimming)) +(dem_dimming * err_before**2 / (dem_before**3)))
#-----------------------------------------------------------------------------------------------------------------------
#Create plot CME_mass = f(sqrt(DEM_ratio))
def linear_func(parameters, x):
    a, b = parameters
    return a * x + b

linear_model = Model(linear_func)
def create_plot(dem_ratio, cme_mass, err_dem, temp):
    # Perform filtering
    mask = cme_mass != 1.4e16 #This is the double events that we exclude
    dem_ratio = dem_ratio[mask]
    cme_mass = cme_mass[mask]
    err_dem = err_dem[mask]

    sqrt_dem_ratio = np.sqrt(dem_ratio)

    #calculate errorbars and input mass for the fit
    errorbars = (2 * cme_mass).copy()
    cme_mass_fit = (2 * cme_mass).copy()
    err_fit_y = cme_mass.copy()

    #the non-halo event (i select it manually because i have few events and i know which one it is)
    errorbars[5] = cme_mass[5]
    cme_mass_fit[5] = 1.5 * cme_mass[5]
    err_fit_y[5] = 0.5 * cme_mass[5]


    data = RealData(sqrt_dem_ratio, cme_mass_fit, sx=err_dem, sy=err_fit_y)
    odr = ODR(data, linear_model, beta0=[0.,1.])
    out=odr.run()
    slope, intercept = out.beta
    fit_line=linear_func(out.beta, sqrt_dem_ratio)

    # Compute R-squared
    residuals = cme_mass_fit - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((cme_mass_fit - np.mean(cme_mass_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute standard error of the slope
    se_slope = out.sd_beta[0]  # Standard error of the slope
    slope = out.beta[0]  # Estimated slope

    # Compute t-statistic
    t_stat = slope / se_slope

    # Degrees of freedom
    n = len(sqrt_dem_ratio)  # Number of data points
    p = len(out.beta)  # Number of parameters (slope and intercept)
    df = n - p  # Degrees of freedom

    # Compute two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    # Plot
    plt.figure(figsize=(16, 9))
    plt.title(f'CME Mass as Function of $\\sqrt{{DEM_{{dimming}}/DEM_{{before}}}}$ for $\\log(T)={temp}$ (K)',fontsize=20)

    # Scatter plot with error bars
    plt.errorbar(sqrt_dem_ratio, cme_mass_fit, xerr=err_dem, yerr=err_fit_y,linestyle=' ', marker=' ', color='red', label='Errorbars')
    plt.scatter(sqrt_dem_ratio, cme_mass, color='blue', label='Data', marker='.')

    # Plot fitted line
    plt.plot(sqrt_dem_ratio, fit_line, color='green', linestyle='--',
             label=(
                 f'Fit: $y = {intercept / 10 ** int(np.floor(np.log10(abs(intercept)))):.3f} \\times 10^{{{int(np.floor(np.log10(abs(intercept))))}}}'
                 f'{slope / 10 ** int(np.floor(np.log10(abs(slope)))):.3f} \\times 10^{{{int(np.floor(np.log10(abs(slope))))}}} x$\n'
                 f'$R^2 = {r_squared:.3f}$, $p = {p_value:.3g}$'))

    # Labels and styling
    plt.ylabel('CME Mass ($g$)',fontsize=16)
    plt.xlabel('$\\sqrt{DEM_{dimming}/DEM_{before}}$',fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#Main program
#load files with DEM data
files2=[
       # "C:\\Users\\ageli\\Desktop\\pythonProject4\\mass\\dem608cor.txt",
        "C:\\Users\\ageli\\Desktop\\pythonProject4\\mass\\dem616cor.txt"
#"C:\\Users\\ageli\\Desktop\\pythonProject4\\mass\\dem625cor.txt"
    ]
temps=['6.16']
#load file with CME masses
cme_masses_file = "C:\\Users\\ageli\\Desktop\\pythonProject4\\event_masses.txt"
def run_the_main_code():
    i=0 #use it as index for the loop
    #Each dem file has the dem info for one temp. So i make a loop for all temps
    for file in files2:
        filename = file
        dem_before, dem_dimming, err_before, err_dimming= read_dem_from_file2(file)  # call read_dem_from_file
        cme_mass = read_mass_file(cme_masses_file) #call masses
        dem_ratio = dem_dimming / dem_before  # create dem_ratio
        #call dem_err
        err_dem = dem_ratio_error(dem_before, dem_dimming, err_before, err_dimming)
        #ensure correct indexing
        sorted_indices = np.argsort(dem_ratio)
        cme_mass_sorted = cme_mass[sorted_indices]
        dem_ratio_sorted = dem_ratio[sorted_indices]
        err_dem_sorted = err_dem[sorted_indices]
        print(cme_mass)
        #make plot
        create_plot(dem_ratio_sorted, cme_mass_sorted, err_dem_sorted, temps[i])  # call create_plot
        i = i+1
run_the_main_code()
