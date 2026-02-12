# Install necessary packages
using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "StatsBase", "Distributions", "SpecialFunctions", "Optim", "BAT"])

println("All packages installed successfully!")

using Plots
using Distributions

# Creating Gaussian for first peak (guessing mean = 93, std = 10)
mu_1 = 93
sigma_1 = 10
peak_1 = Normal(mu_1, sigma_1)

# Defining domain, range for plot
x_range = range(0, 255, length = 51)
peak1_pdf = pdf.(peak_1, x_range)

# Creating Gaussian for Higgs signal (second peak) (guessing mean = 125.3, std = 10)
mu_2 = 125.3
sigma_2 = 15
peak_2 = Normal(mu_2, sigma_2)
peak2_pdf = pdf.(peak_2, x_range)

# Plotting
plot(x_range,
     peak1_pdf + peak2_pdf, 
     label="Peak1 + Signal", 
     xlabel="4-lepton Invariant Mass m_4l (GeV)", 
     ylabel="Events/5GeV", 
     title="Peak 1 & Higgs Signal Gaussian Distribution",
     linewidth = 2)

