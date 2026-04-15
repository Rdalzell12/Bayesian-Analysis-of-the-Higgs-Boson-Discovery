import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
using CSV, DataFrames

#Load in csv file
df = CSV.read("13TeV_diphoton_extracted.csv", DataFrame)

#Checking how many initial events are in the dataset
println("Loaded ", nrow(df), " events")


#Convert MeV to GeV for photon transverse momentum and photon energy
for i in 1:2
    df[!, Symbol("photon_pt_$i")] /= 1000
    df[!, Symbol("photon_E_$i")]  /= 1000
end


#Initiate cutflow tracking
cutflow = Dict{String,Int}()
cutflow["Initial"] = nrow(df)

#MC cut - extracting only MC Data from CERN
mc_mask = (df.dataset .!= "data")
df = df[mc_mask, :]
cutflow["mc"] = nrow(df)


#Trigger cut - data must agree with CERN's photon criteria  
trigger_mask = (df.trigP_1 .== true) .& (df.trigP_2 .== true)
df = df[trigger_mask, :]
cutflow["Trigger"] = nrow(df)


#Photon kinematic cuts

#Pseudorapidity cut - data must have pseudorapidity less than 2.37 
eta_cut = (abs.(df.photon_eta_1) .< 2.37) .& (abs.(df.photon_eta_2) .< 2.37)

#Crack veto - gets rid of photons with energy in the energy 'crack' caused by detector geometery 1.37 < eta < 1.52
crack_veto = .!(
    ((abs.(df.photon_eta_1) .> 1.37) .& (abs.(df.photon_eta_1) .< 1.52)) .|
    ((abs.(df.photon_eta_2) .> 1.37) .& (abs.(df.photon_eta_2) .< 1.52))
)

#Transverse momentum cut - photons must have transverse momentum more than 25
pt_cut = (df.photon_pt_1 .> 40) .& (df.photon_pt_2 .> 30)

#Combining all kinematic cuts
mask = (eta_cut .& crack_veto .& pt_cut)
df = df[mask, :]
cutflow["Photon Kinematics"] = nrow(df)


#Photon identification - extracting only data that fits CERN's tight photon criteria
id_cut = (df.photon_isTightID_1.==1) .& (df.photon_isTightID_2.==1)
df = df[id_cut, :]
cutflow["Photon Tight ID"] = nrow(df)


#Photon isolation
track_iso =
(df.photon_ptcone30_1 ./ df.photon_pt_1 .< 0.05) .&
(df.photon_ptcone30_2 ./ df.photon_pt_2 .< 0.05)

calo_iso =
(df.photon_etcone20_1 ./ df.photon_pt_1 .< 0.065) .&
(df.photon_etcone20_2 ./ df.photon_pt_2 .< 0.065)

iso_cut = track_iso .& calo_iso
df = df[iso_cut, :]
cutflow["Photon Isolation"] = nrow(df)


#find diphoton mass
#equation: m^2=(E1​+E2​)^2−∣p​1​+p​2​∣^2
px1 = df.photon_pt_1 .* cos.(df.photon_phi_1)
py1 = df.photon_pt_1 .* sin.(df.photon_phi_1)
pz1 = df.photon_pt_1 .* sinh.(df.photon_eta_1)

px2 = df.photon_pt_2 .* cos.(df.photon_phi_2)
py2 = df.photon_pt_2 .* sin.(df.photon_phi_2)
pz2 = df.photon_pt_2 .* sinh.(df.photon_eta_2)

E = df.photon_E_1 .+ df.photon_E_2

px = px1 .+ px2
py = py1 .+ py2
pz = pz1 .+ pz2

df.m_gg = sqrt.(max.(E.^2 .- px.^2 .- py.^2 .- pz.^2, 0))

println("Minimum Invariant Mass = ", minimum(df.m_gg))
println("Maximum Invariant Mass = ", maximum(df.m_gg))


#ET/mγγ cuts - extracting photons with correct ratio of photon's transverse energy to diphoton invariant mass
ratio_cut =
(df.photon_pt_1 ./ df.m_gg .> 0.35) .&
(df.photon_pt_2 ./ df.m_gg .> 0.25)

df = df[ratio_cut, :]
cutflow["ET/mγγ"] = nrow(df)


#Diphoton mass window cut - extracting only a reasonable range of mass values
mass_cut = (df.m_gg .> 105) .& (df.m_gg .< 160)
df = df[mass_cut, :]
cutflow["Mass Window"] = nrow(df)


#Final weighted yield
signal_yield = sum(df.event_weight)

println("Final weighted yield = ", signal_yield)


#Making cutflow table
println("Cutflow - events left after each cut:")

cut_order = [
"mc",
"Initial",
"Trigger",
"Photon Kinematics",
"Photon Tight ID",
"Photon Isolation",
"ET/mγγ",
"Mass Window", 
]

for c in cut_order
    println(rpad(c,25), cutflow[c])
end


#Plot of resulting data with line at the Higgs peak
using Plots

histogram(
    df.m_gg,
    bins=100,
    yscale=:log10,
    xlabel="mγγ (GeV)",
    ylabel="Number of Events",
    label="Diphoton Mass",
    title="Diphoton Invariant Mass - Only MC",
    color=:pink
)
    vline!([125], label="Higgs Mass", linewidth=2)


println("Final Dataset Shape: ", size(df) )


#Exporting final dataset to a CSV
CSV.write(joinpath(homedir(), "separated_and_cut_mc_13TeV.csv"), df)