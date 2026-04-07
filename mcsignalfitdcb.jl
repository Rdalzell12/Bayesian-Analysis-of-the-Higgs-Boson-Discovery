using CSV
using Distributions
using DataFrames
using BAT
using Plots
using DensityInterface
using Measures
using Optim
using IntervalSets

data_dir = joinpath(@__DIR__, "higgs_fit_data")
df_mc_sig = CSV.read(joinpath(data_dir, "higgs_fit_histograms.csv"), DataFrame)

bin_centers = Float64.(df_mc_sig.bin_center)
mc_sig = Float64.(df_mc_sig.signal_expected)
bin_width = Float64(df_mc_sig.bin_width[1])

function dcb(x, mu, sigma, alphaL, nL, alphaR, nR)
    if sigma <= 0 || alphaL <= 0 || alphaR <= 0 || nL <= 1 || nR <= 1
        return 0.0
    end

    t = (x - mu) / sigma

    if t < -alphaL
        AL = (nL / alphaL)^nL * exp(-0.5 * alphaL^2)
        BL = nL / alphaL - alphaL
        return AL * (BL - t)^(-nL)
    elseif t <= alphaR
        return exp(-0.5 * t^2)
    else
        AR = (nR / alphaR)^nR * exp(-0.5 * alphaR^2)
        BR = nR / alphaR - alphaR
        return AR * (BR + t)^(-nR)
    end
end

function eval_dcb(bin_centers, pars)
    mu, sigma, alphaL, nL, alphaR, nR = pars
    [dcb(x, mu, sigma, alphaL, nL, alphaR, nR) for x in bin_centers]
end

function normalize_shape(vals, bin_width)
    s = sum(vals) * bin_width
    if !isfinite(s) || s <= 0
        return nothing
    end
    vals ./ s
end

function logL_sig_only(p, bin_centers, mc_sig, bin_width)
    pars = (
        Float64(p.mean),
        Float64(p.sigma),
        Float64(p.alphaL),
        Float64(p.nL),
        Float64(p.alphaR),
        Float64(p.nR)
    )

    shape = eval_dcb(bin_centers, pars)
    shape_norm = normalize_shape(shape, bin_width)

    if shape_norm === nothing || any(.!isfinite.(shape_norm))
        return -Inf
    end

    lambda = p.mu_sig .* shape_norm .* bin_width

    if any(lambda .<= 0) || any(.!isfinite.(lambda))
        return -Inf
    end

    sigma_mc = sqrt.(max.(mc_sig, 1e-6))
    vals = logpdf.(Normal.(lambda, sigma_mc), mc_sig)

    if any(.!isfinite.(vals))
        return -Inf
    end

    sum(vals)
end

total_sig = sum(mc_sig)
peak_guess = bin_centers[argmax(mc_sig)]

prior = BAT.NamedTupleDist(
    mu_sig = LogNormal(log(max(total_sig, 1.0)), 0.5),
    mean = Uniform(peak_guess - 5.0, peak_guess + 5.0),
    sigma = Uniform(0.5, 5.0),
    alphaL = Uniform(0.5, 5.0),
    nL = Uniform(1.1, 20.0),
    alphaR = Uniform(0.5, 5.0),
    nR = Uniform(1.1, 20.0)
)

likelihood = logfuncdensity(
    p -> logL_sig_only(p, bin_centers, mc_sig, bin_width)
)

posterior = PosteriorMeasure(likelihood, prior)

samples = bat_sample(
    posterior,
    MCMCSampling(
        mcalg = RandomWalk(),
        nsteps = 100000,
        nchains = 4
    )
).result

best = bat_findmode(posterior).result

println("Signal best-fit parameters:")
println("mu_sig  = ", round(best.mu_sig, digits = 4))
println("mean    = ", round(best.mean, digits = 4))
println("sigma   = ", round(best.sigma, digits = 4))
println("alphaL  = ", round(best.alphaL, digits = 4))
println("nL      = ", round(best.nL, digits = 4))
println("alphaR  = ", round(best.alphaR, digits = 4))
println("nR      = ", round(best.nR, digits = 4))

pars_best = (
    Float64(best.mean),
    Float64(best.sigma),
    Float64(best.alphaL),
    Float64(best.nL),
    Float64(best.alphaR),
    Float64(best.nR)
)

shape_best = eval_dcb(bin_centers, pars_best)
shape_best = normalize_shape(shape_best, bin_width)
lambda_best = best.mu_sig .* shape_best .* bin_width

x_fine = range(minimum(bin_centers), maximum(bin_centers), length=500)
dx_fine = step(x_fine)

y_fine_raw = eval_dcb(x_fine, pars_best)
y_fine_norm = normalize_shape(y_fine_raw, dx_fine)
y_fine = best.mu_sig .* y_fine_norm .* bin_width

plot_1 = plot(
    bin_centers,
    mc_sig,
    seriestype = :scatter,
    yerr = sqrt.(max.(mc_sig, 1e-6)),
    label = "Signal MC",
    xlabel = "Invariant Mass [GeV]",
    ylabel = "Counts",
    title = "Signal MC Fit with Double Crystal Ball",
    color = "black",
    framestyle = "box"
)

plot!(
    x_fine,
    y_fine,
    linewidth = 2,
    label = "DCB best fit",
    color = "red"
)

display(plot_1)

println("\nfloated variables:")
println("mu_sig = ", best.mu_sig)
println("mean = ", best.mean)
println("sigma = ", best.sigma)
println("alphaL = ", best.alphaL)
println("nL = ", best.nL)
println("alphaR = ", best.alphaR)
println("nR = ", best.nR)

#reduced chi-square analysis on double crystal ball
observed = mc_sig
expected = lambda_best

function chi_squared(observed, expected)
    return sum((observed .- expected).^2 ./ expected)
end

chi2 = chi_squared(observed, expected)
nbins = length(observed)
nparams = 7
dof = nbins - nparams
reduced_chi2 = chi2 / dof
p_value = 1 - cdf(Chisq(dof), chi2)

println("Chi-squared value: ", chi2)
println("Reduced chi-squared value: ", reduced_chi2)
println("p value: ", p_value)

if reduced_chi2 >= 0.5 && reduced_chi2 <= 1.5
    println("acceptable reduced chi-squared value")
else
    println("unacceptable reduced chi-squared value")
end

if p_value >= 0.05 && p_value <= 0.95
    println("acceptable p value")
else
    println("unaccetpable p value")
end