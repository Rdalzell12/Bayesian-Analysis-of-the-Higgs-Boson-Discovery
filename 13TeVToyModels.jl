#using Pkg
#Pkg.add(["CSV", "DataFrames", "StatsBase", "BAT", "Distributions", 
#        "Optim", "Plots", "IntervalSets", "DensityInterface", 
#        "Measures"])

module HiggsAnalysis

using CSV
using DataFrames
using StatsBase
using BAT
using Distributions
using Optim
using Plots
using IntervalSets
using DensityInterface
using Measures

export log_likelihood, fit_signal, fit_background, fit_data_observed, fit_bin_centers, fit_bin_width

# 1) Names and DataFrames
# 1a) Full spectrum histogram
data_dir = joinpath(@__DIR__, "higgs_fit_data")

df_full = CSV.read(
    joinpath(data_dir, "full_spectrum_histograms.csv"),
    DataFrame
)

full_bin_centers = df_full.bin_center
full_bin_width = df_full.bin_width
full_data_observed = df_full.data_observed
full_error = df_full.data_stat_error
full_signal = df_full.signal_expected
full_background = df_full.background_expected

# 1b) Fit Window histogram
df_fit = CSV.read(joinpath(data_dir, "higgs_fit_histograms.csv"), DataFrame)

fit_bin_centers = df_fit.bin_center
fit_bin_width = df_fit.bin_width
fit_data_observed = df_fit.data_observed
fit_error = df_fit.data_stat_error
fit_signal = df_fit.signal_expected
fit_background = df_fit.background_expected


# 2) MC + Data plot, full spectrum histogram
plot_full = plot(
    full_bin_centers,
    full_data_observed,
    seriestype = :scatter,
    yerr = sqrt.(full_data_observed),
    label = "Observed Data",
    xlabel = "m4l [GeV/c^2]",
    ylabel = "Events / $(Int(full_bin_width[1])) GeV",
    title = "H -> ZZ* -> 4l Full Spectrum (ATLAS Open Data)",
    framestyle = :box,
    legend = :topleft,
    grid = false,
    markersize = 3,
    size = (800, 600),
    color = :black,
    xlim = (80, 250),
    left_margin = 7mm,
    bottom_margin = 3mm,
    right_margin = 4mm,
    top_margin = 4mm
)

bar!(full_bin_centers,
     full_background,
     bar_width = full_bin_width * .90,
     label = "Background",
     alpha = 0.8,
     color = "turquoise4",
)

bar!(full_bin_centers,
     full_signal,
     bin_width = full_bin_width * .90,
     label = "Signal",
     color = "red",
     alpha = 0.8
)

scatter!(full_bin_centers,
         full_data_observed,
         yerr = sqrt.(full_data_observed),
         label = "",
         color = "black",
         markersize = 5
)

vspan!([110, 135], color = "orange", alpha = 0.1, label = "Higgs Signal Region")

display(plot_full)



# For the physics model, n_i ~ poisson(λ_i)
# and λ_i(μ) = μ(s_i) + b_i where s_i, b_i are expected signal, background respectively.
# Log likelihood instead of likelihood for computation efficiency, no underflow to 0, 
# MCMC compatibility, and additive convenience.
function log_likelihood(mu, fit_data_observed, fit_signal, fit_background)
    lambda = mu .* fit_signal .+ fit_background
    if any(lambda .<= 0)
        return -Inf
    end
    return sum(logpdf.(Poisson.(lambda), fit_data_observed))
end

# Prior distribution!!!
prior = BAT.NamedTupleDist(
    mu = Uniform(0, 5)
)

log_likelihood_density = logfuncdensity() do params
    log_likelihood(params.mu,
                   fit_data_observed,
                   fit_signal,
                   fit_background)
end

# Posterior
posterior = PosteriorDensity(log_likelihood_density, prior)

samples = bat_sample(
    posterior,
    MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 4)
    # could also use 'mcalg = HamiltonianMonteCarlo'
).result

println("BAT STATS...")
println("coming soon to a town near you!!!!!!")
bat_report(samples)
display(bat_report(samples))

samples_mode = mode(samples)
best_fit = bat_findmode(
    posterior
).result

println("\nSample Mode: ", round(samples_mode.mu, digits=5))
println("Best Fit Signal Strength: ", round(best_fit.mu, digits=5))

# Bayesian stats
mu_samples = [s.mu for s in samples.v]

mu_mean = mean(mu_samples)
mu_std = std(mu_samples)
mu_median = median(mu_samples)

mu_lower_68 = quantile(mu_samples, 0.16)
mu_upper_68 = quantile(mu_samples, 0.84)
mu_lower_95 = quantile(mu_samples, 0.025)
mu_upper_95 = quantile(mu_samples, 0.975)


# Probability Density Function
pdf_posterior = histogram(mu_samples,
    bins = 60,
    # normalize = :pdf,
    label = "Probability of μ given data",
    xlabel = "Signal strength (μ)",
    ylabel = "Probability Density",
    title = "Probability Density by Signal Strength (μ)",
    color = :turquoise4,
    framestyle = :box,
    size = (900, 600),
    fillalpha = 1.0,
    left_margin = 5mm,
    bottom_margin = 3mm,
    right_margin = 3mm,
    top_margin = 3mm
)

vline!([1.0], label = "SM (μ=1)", lw = 2, color = :green, linestyle = :dash)
vline!([0.0], label = "No signal (μ=0)", lw = 2, color = :red, linestyle = :dash)
vline!([mu_mean], label = "Julia: μ=$(round(mu_mean, digits=2))", lw = 2, linestyle = :dash, color = :yellow)

# Shade 68% credible interval
vspan!([mu_lower_68, mu_upper_68], alpha=0.2, color=:steelblue, label="68% CI")

display(pdf_posterior)


# Best fit model for the higgs window
expected_bestfit = best_fit.mu .* fit_signal .+ fit_background

p_fit = plot(
    fit_bin_centers,
    fit_data_observed,
    seriestype = :scatter,
    yerr = sqrt.(fit_data_observed),
    label = "Data",
    xlabel = "m4l [GeV]",
    ylabel = "Events / 2 GeV",
    title = "H -> ZZ* -> 4l: Bayesian fit (110-140 GeV)",
    framestyle = :box,
    grid = false,
    legend = :topright,
    markersize = 5,
    color = :black,
    size = (900, 600),
    left_margin = 5mm,
    bottom_margin = 3mm,
    right_margin = 2mm
)

# background
bar!(fit_bin_centers,
     collect(fit_background), 
     bar_width = fit_bin_width * 0.9,
     label = "Background (MC)", 
     color = :turquoise4,
     alpha = 0.5)

# Scaled μ with best fit
bar!(fit_bin_centers,
     best_fit.mu .* collect(fit_signal), 
     bar_width = fit_bin_width * 0.9,
     bottom = collect(fit_background),
     label = "Signal * μ=$(round(best_fit.mu, digits=2))", 
     color = :red,
     alpha = 0.7)

# Overlay total expectation as a line
plot!(fit_bin_centers, 
      expected_bestfit, 
      label = "Total fit (μ*S + B)",
      lw = 2, 
      color = :black,
      linestyle = :solid,
      marker = :circle,
      markersize = 3)

# Also show SM prediction (μ=1)
expected_SM = fit_signal .+ fit_background
plot!(fit_bin_centers, 
      expected_SM, 
      label = "SM (μ=1)",
      lw = 2, 
      color = :green,
      linestyle = :dash)

# Scatter data on top
scatter!(fit_bin_centers, 
         fit_data_observed,
         yerr = sqrt.(fit_data_observed),
         label = "",
         color = :black,
         markersize = 7)

display(p_fit)

# Python frequentist results to compare
df_meta = CSV.read(joinpath(data_dir, "metadata.csv"), DataFrame)

py_mu = df_meta.python_best_fit_mu[1]
py_mu_err = df_meta.python_mu_error[1]
py_mu_significance = df_meta.python_significance[1]


# Bayesian-Julia comparison with Frequentist-python
println("\n" * "="^30)
println("DATA REPORT")
println("="^30)

println("\n----Full Spectrum of Data, 80-250 [GeV]----")
println("Full spectrum data count: ", Int(sum(full_data_observed)))
println("Bin width: ", full_bin_width[1])
println("Bin count: ", length(full_bin_centers))
println("Template spectrum signal count (MC): ", round(sum(full_signal), digits=3))
println("Template background count (MC): ", round(sum(full_background), digits=3))

println("\n----Higgs Signal Fit Window, 110-140 [GeV]----")
println("Higgs signal fit window data count: ", Int(sum(fit_data_observed)))
println("Bin width: ", fit_bin_width[1])
println("Bin count: ", length(fit_bin_centers))
println("Template signal count (MC): ", round(sum(fit_signal), digits=3))
println("Template background count (MC): ", round(sum(fit_background), digits=3))

println("\n----Results of Bayesian Fit in Julia----")
println("μ mean: ", round(mu_mean, digits=2), " ± ", round(mu_std, digits=2))
println("μ median: ", round(mu_median, digits=2))
println("1σ (68%) credible interval: ", "(", round(mu_lower_68, digits=2), ", ", 
        round(mu_upper_68, digits=2), ")")
println("2σ (95%) credible interval: ", "(", round(mu_lower_95, digits=2), ", ", 
        round(mu_upper_95, digits=2), ")")
println("Bayesian significance: ", round(mu_mean / mu_std, digits=2), "σ")

println("\n----Results of Frequentist Fit in Python----")
println("μ mean: ", round(py_mu, digits=2), " ± ", round(py_mu_err, digits=2))
println("Frequentist significance: ", round(py_mu_significance, digits=2), "σ")

println("\n----Agreement Test----")
println("Difference in results:|(Bayes σ) - (Frequentist σ)| = ", 
        round(abs(mu_mean - py_mu), digits=2))
println("The results agree.")
println("Difference in significance: ", 
        round(abs(mu_mean / mu_std - py_mu_significance), digits=2))

# ===== Toy Experiments: Generate signal + background toys =====

function generate_toy_experiment(signal, background, mu)
    # Generate Poisson counts for each bin based on signal + background
    expected_counts = mu .* signal .+ background
    toy_counts = [rand(Poisson(λ)) for λ in expected_counts]
    return toy_counts
end

# Generate 10 toy experiments using fit window data
toys = [generate_toy_experiment(fit_signal, fit_background, best_fit.mu) for _ in 1:10]

# Plot toy experiments
p_toys = plot(
    xlabel = "m4l [GeV]",
    ylabel = "Events / 2 GeV",
    title = "Toy Experiments: Signal + Background",
    framestyle = :box,
    size = (900, 600),
    legend = :topright,
    left_margin = 5mm,
    bottom_margin = 3mm,
    right_margin = 2mm
)

# Show all 10 toy experiments
for (i, toy) in enumerate(toys)
    bar!(p_toys, fit_bin_centers, toy, 
          label = "Toy $i", 
          alpha = 0.6,)
end

# Overlay the expected signal + background (μ=1)
expected_nominal = fit_signal .+ fit_background
plot!(p_toys, fit_bin_centers, expected_nominal,
      label = "Expected (μ=1, S+B)",
      seriestype = :steppost,
      color = :green,
      lw = 3,
      linestyle = :dash)

# Overlay best fit expectation
expected_bestfit = best_fit.mu .* fit_signal .+ fit_background
plot!(p_toys, fit_bin_centers, expected_bestfit,
      label = "Best Fit (μ=$(round(best_fit.mu, digits=2)), μS+B)",
      seriestype = :steppost,
      color = :red,
      lw = 3)

display(p_toys)

display(p_toys)

end