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
df_mc_bkg = CSV.read(joinpath(data_dir, "background_template.csv"), DataFrame)

bin_centers = df_mc_bkg.bin_center
mc_bkg = df_mc_bkg.background
bin_width = 2.0

bernstein_basis(t::Real, k::Int, n::Int) =
    binomial(n, k) * t^k * (1 - t)^(n - k)

function bernstein_poly(t::Real, coeffs::NTuple{4,Float64})
    c0, c1, c2, c3 = coeffs
    return c0 * bernstein_basis(t, 0, 3) +
           c1 * bernstein_basis(t, 1, 3) +
           c2 * bernstein_basis(t, 2, 3) +
           c3 * bernstein_basis(t, 3, 3)
end

to_unit_interval(x, x_min, x_max) =
    (x - x_min) / (x_max - x_min)

function eval_bernstein(bin_centers, coeffs, bin_width)
    n = length(bin_centers)
    B = zeros(Float64, n)
    x_min = minimum(bin_centers) - bin_width / 2
    x_max = maximum(bin_centers) + bin_width / 2

    for i in 1:n
        t = to_unit_interval(bin_centers[i], x_min, x_max)
        B[i] = bernstein_poly(t, coeffs)
    end

    return B
end

function normalize_bernstein(B, bin_width)
    norm = sum(B) * bin_width
    if norm <= 0 || !isfinite(norm)
        return nothing
    end
    return B ./ norm
end

coeffs = (1.0, 1.0, 1.0, 1.0)

B = eval_bernstein(bin_centers, coeffs, bin_width)
p = normalize_bernstein(B, bin_width)

println(sum(p) * bin_width)

function logL_bkg_only(p, bin_centers, mc_bkg, bin_width)
    coeffs = (Float64(p.c0), Float64(p.c1), Float64(p.c2), Float64(p.c3))
    B = eval_bernstein(bin_centers, coeffs, bin_width)
    p_norm = normalize_bernstein(B, bin_width)

    if p_norm === nothing || any(.!isfinite.(p_norm))
        return -Inf
    end

    lambda = p.mu_bkg .* p_norm .* bin_width

    if any(lambda .<= 0) || any(.!isfinite.(lambda))
        return -Inf
    end

    sigma = sqrt.(max.(mc_bkg, 1e-6))
    vals = logpdf.(Normal.(lambda, sigma), mc_bkg)

    if any(.!isfinite.(vals))
        return -Inf
    end

    return sum(vals)
end

#LogNormal(mean, log scale)
#Uniform(lower bound, upper bound)
#Normal(mean, std)
prior = BAT.NamedTupleDist(
    mu_bkg = Uniform(0.0, 10.0),
    c0 = Uniform(0.0, 10),
    c1 = Uniform(0.0, 10),
    c2 = Uniform(0.0, 10),
    c3 = Uniform(0.0, 10)
)

likelihood = logfuncdensity(
    p -> logL_bkg_only(p, bin_centers, mc_bkg, bin_width)
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

println("Background best-fit mu_bkg = ", round(best.mu_bkg, digits = 4))

coeffs_best = (
    Float64(best.c0),
    Float64(best.c1),
    Float64(best.c2),
    Float64(best.c3)
)

b_best = eval_bernstein(bin_centers, coeffs_best, bin_width)
p_best = normalize_bernstein(b_best, bin_width)
lambda_best = best.mu_bkg .* p_best .* bin_width

plot_1 = plot(
    bin_centers,
    mc_bkg,
    seriestype = :scatter,
    yerr = sqrt.(mc_bkg),
    label = "MC background",
    title = "Background Counts vs. Invariant Mass",
    xlabel = "Invariant Mass [GeV]",
    ylabel = "Counts",
    color = "black",
    framestyle = "box",
    bottom_margin = 5mm,
    left_margin = 5mm,
    top_margin = 5mm,
    right_margin = 5mm
)

plot!(
    bin_centers,
    lambda_best,
    linewidth = 2,
    label = "best fit",
    color = "green"
)

display(plot_1)

println("\nfloated variables: ")
println("mu_bkg = ", best.mu_bkg) 
println("c0 = ", best.c0)
println("c1 = ", best.c1)
println("c2 = ", best.c2)
println("c3 = ", best.c3)

#reduced chi-square analysis on bernstein fit
observed = mc_bkg
expected = lambda_best

function chi_squared(observed, expected)
    return sum((observed .- expected).^2 ./ expected)
end

chi2 = chi_squared(observed, expected)
nbins = length(observed)
nparams = 5
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