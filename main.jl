
using Base.Threads
using Plots, Pipe, DataFrames, MixedModels
using LaTeXStrings, StatsBase, CategoricalArrays
using CSV, JLD
using Distributions, StableRNGs
using LambertW, Optim

####################################################################################
#   FUNCTIONS THAT WILL BE USED THROUGHOUT SCRIPT
####################################################################################

"""
    tpeak(a, b)

    Arguments: 
        a: decay time constant 
        b: other time constant 
    
    Returns: 
        Time to peak magnitude of effect 
"""
function ftpeak(a,b) 
    if a > b 
        out = - ((a*b)/(a-b)) * log(b/a)
    elseif a == b 
        out = b 
    else 
        out = NaN 
    end 
    return out 
end

"""
    tpeak_to_b(t,a)

    Arguments:
        t: Time to peak response 
        a: Decay time 
    
    Returns:
        b: Other time constant

"""
function tpeak_to_b(t, a)
    return -t/lambertw(-(exp(-t/a)*t)/a)
end 

"""
    Gnorm(t,p)

    This is the primary ketamine response timecourse function. Given 
        some time `t>0` and parameters `p=[g,a,b]`, where `g` is the 
        response magnitude, `a` is the decay time, and `b` is a
        second time constant, return the overall response.
"""
function Gnorm(t, p) 
    g, a, b = p
    if a ≈ b 
        out = g .* t .* exp.(1 .- (t/b))/b
    else
        out = g*(exp.(-t/a) - exp.(-t/b))/(-(b/a)^(a/(a-b)) + (b/a)^(b/(a-b)))
    end
    return out
end

"""
    sample_trajectory(g,a,b,σ,trange)

    Arguments:
        g: The magnitude of ketamine response 
        a: Decay time constant 
        b: Time constant 
        σ: Gaussian noise around the mean 
        trange: `start:dt:end` specifying the time points to simulate 
"""
sample_trajectory(g,a,b,σ,trange) = @pipe trange |> 
    collect |> 
    (_, map(x-> x + rand(Normal(0,σ)), Gnorm(_,[g,a,b])))

"""
    sample_fit(g, a, b, σ, trange; maxiters=10)

    Arguments:
        g: The magnitude of ketamine response 
        a: Decay time constant 
        b: Time constant 
        σ: Gaussian noise around the mean 
        trange: `start:dt:end` specifying the time points to simulate
        maxiters: maximum number of optimization iterations
"""
function sample_fit(g, a, b, σ, trange; maxiters=10)
    done = false; n_iters = 0; out = nothing
    while !done
        n_iters += 1
        x, y = sample_trajectory(g, a, b, σ, trange)
        function loss(p)
            if (p[3] > p[2]) | (p[1] < -1) | (p[1] > 1)
                out = 1e3
            else 
                out = sum((y .- Gnorm(x, p)).^2)
            end 
            return out
        end
        
        res = optimize(loss, [-1., 0.01, 0.01], [1., Inf, Inf], [0., 2., 1.])
        
        if (n_iters >= maxiters) | Optim.converged(res)
            done = true 
            out = res
        end 
    end
    return out
end

mutable struct ExperimentalParams 
    n_runs 
    g_min 
    g_max 
    b_min 
    b_max 
    adelta_min 
    adelta_max 
    σ
    tmin 
    tmax
    dt
    rng

    function ExperimentalParams(n_runs, g_min, g_max, b_min, b_max, adelta_min, adelta_max, σ, tmin, tmax, dt, random_state)
        new(n_runs, g_min, g_max, b_min, b_max, adelta_min, adelta_max, σ, tmin, tmax, dt, StableRNG(random_state))
    end
end


"""
    make_paramlist(par)

    Arguments: 
        par: ExperimentalParams struct 

    Returns: 
        Matrix with the function parameters in the following order 
            [g, a, b, σ]
"""
function make_paramlist(par)
    g_vals = rand(par.rng, Uniform(par.g_min, par.g_max), par.n_runs) 
    b_vals = rand(par.rng, Uniform(par.b_min, par.b_max), par.n_runs)
    a_vals = b_vals .+ rand(par.rng, Uniform(par.adelta_min, par.adelta_max), par.n_runs)
    return @pipe [[g_vals[i], a_vals[i], b_vals[i], par.σ] for i ∈ 1:par.n_runs] |> 
        mapreduce(permutedims, vcat, _)
end

"""
    run_experiment(par)

    Arguments: 
        par: ExperimentalParams struct 
    
    Returns:
        - Matrix with true parameters [g,a,b]
        - Matrix with parameters recovered from optimization [g,a,b]
"""
function run_experiment(par)
    paramlist = make_paramlist(par)
    res = @pipe [sample_fit(paramlist[i,:]..., par.tmin:par.dt:par.tmax) |> Optim.minimizer for i ∈ 1:size(paramlist, 1)] |> 
        mapreduce(permutedims, vcat, _)
    return paramlist[:,1:3], res
end

median_absolute_deviation(X, Y) = (vec ∘ median)(abs.(X .- Y), dims=1)
percentile_absolute_deviation(X, Y; p=50) = @pipe abs.(X .- Y) |> [percentile(_[:,i], p) for i ∈ 1:size(_, 2)]

function plot_fits(par, X, Y; alpha=0.5)
    f1 = plot(-par.g_min:par.g_max, x->x, c=:black, ls=:dash, legend=false, xlabel=L"g", ylabel=L"\hat{g}")
    f1 = scatter!(X[:,1], Y[:,1], c=:black, alpha=alpha)
    f2 = plot(0:(par.b_max + par.adelta_max), x->x, c=:black, ls=:dash, legend=false, xlabel=L"a", ylabel=L"\hat{a}", xlims=[0,par.b_max + par.adelta_max], ylims=[0,par.b_max + par.adelta_max])
    f2 = scatter!(X[:,2], Y[:,2], c=:black, alpha=alpha)
    f3 = plot(0:par.b_max, x->x, c=:black, ls=:dash, legend=false, xlabel=L"b", ylabel=L"\hat{b}", xlims=[0,par.b_max], ylims=[0,par.b_max])
    f3 = scatter!(X[:,3], Y[:,3], c=:black, alpha=alpha)
    f4 = scatter(Y[:,1], Y[:,2], xlabel=L"\hat{g}", ylabel=L"\hat{a}", c=:black, legend=false, xlims=[par.g_min, par.g_max], ylims=[0, par.b_max + par.adelta_max], alpha=alpha)
    f5 = scatter(Y[:,1], Y[:,3], xlabel=L"\hat{g}", ylabel=L"\hat{b}", c=:black, legend=false, xlims=[par.g_min, par.g_max], ylims=[0,par.b_max], alpha=alpha)
    f6 = scatter(Y[:,2], Y[:,3], xlabel=L"\hat{a}", ylabel=L"\hat{b}", c=:black, legend=false, xlims=[0, par.b_max+par.adelta_max], ylims=[0,par.b_max], alpha=alpha)
    return plot(f1, f2, f3, f4, f5, f6, layout = @layout [a b c; d e f])
end

####################################################################################
#   FIGURE 1: RESPONSE FUNCTION AND TIME TO PEAK
####################################################################################
println("PLOTTING FIGURE 1: RESPONSE FUNCTION AND TIME TO PEAK")
ls = [:solid, :dash, :dot]
global g_fig = plot(xlabel=L"t", ylabel=L"\Delta", legend=:bottomright)
for (i,g) ∈ enumerate([-0.8, -0.5, 0.2])
    global g_fig = plot!(0:0.1:10, t->Gnorm(t, [g, 2, 1]), c=:black, ls=ls[i], label=L"g="*"$g")
end
g_fig = annotate!(-3.5, 0.2, Plots.text("A", 20, :left))

global a_fig = plot(xlabel=L"t", ylabel=L"\Delta", legend=:bottomright)
for (i,a) ∈ enumerate([3,4,5])
    global a_fig = plot!(0:0.1:10, t->Gnorm(t, [-1, a, 1]), c=:black, ls=ls[i], label=L"a="*"$a")
end
a_fig = annotate!(-3.5, 0., Plots.text("B", 20, :left))

global b_fig = plot(xlabel=L"t", ylabel=L"\Delta", legend=:bottomright)
for (i,b) ∈ enumerate([1,2,3])
    global b_fig = plot!(0:0.1:10, t->Gnorm(t, [-1, 3, b]), c=:black, ls=ls[i], label=L"b="*"$b")
end
b_fig = annotate!(-3.5, 0., Plots.text("C", 20, :left))

tpeak_fig = contourf(0:0.01:5, 0:0.01:5, (x,y)->tpeak(x,y), xlabel=L"a", ylabel=L"b", colorbar_title=L"t_{peak}")
tpeak_fig = annotate!(-2.1, 5., Plots.text("D", 20, :left))

fig1 = plot(g_fig, a_fig, b_fig, tpeak_fig, layout=@layout [a b; c d])
savefig(fig1, "Figure1.pdf")

####################################################################################
#   PLOT OVERALL FIT
####################################################################################
println("PLOTTING FIGURE 2: OVERALL FIT")
par = ExperimentalParams(500,-1,1,0,5,0.01,4,0.001,0,21,1,2335) 
X, Y = run_experiment(par)
fig = plot_fits(par, X, Y; alpha=0.2)
savefig(fig, "Figure2.pdf")

####################################################################################
#   FIGURE 3: PLOT SENSITIVITY TO NOISE
####################################################################################
println("PLOTTING FIGURE 3: SENSITIVITY TO NOISE")
sigma_range = 0.001:0.01:0.25 |> collect
mad_res = @pipe [(
    @pipe ExperimentalParams(100,-1,1,0,5,0.01,4,σ,0,10,1,235) |> 
        run_experiment |> 
        [σ; median_absolute_deviation(_...); 
            percentile_absolute_deviation(_...; p=25); 
            percentile_absolute_deviation(_...; p=75)]
    ) for σ ∈ sigma_range] |> 
    mapreduce(permutedims, vcat, _)

traj_fig1 = plot(xlabel=L"t", ylabel=L"\Delta", legend=false)
for i ∈ 1:50
    traj_fig1 = plot!(sample_trajectory(-0.5, 2, 1, 0.01, 0:1:10)..., c=:black, alpha=0.5)
end 

traj_fig2 = plot(xlabel=L"t", ylabel=L"\Delta", legend=false)
for i ∈ 1:50
    traj_fig2 = plot!(sample_trajectory(-0.5, 2, 1, 0.1, 0:1:10)..., c=:black, alpha=0.5)
end 

traj_fig3 = plot(xlabel=L"t", ylabel=L"\Delta", legend=false)
for i ∈ 1:50
    traj_fig3 = plot!(sample_trajectory(-0.5, 2, 1, 0.25, 0:1:10)..., c=:black, alpha=0.5)
end 

traj_fig1

k_list = 2:4 |> collect 
ylabels = [L"|\hat{g} - g|", L"|\hat{a} - a|", L"|\hat{b} - b|"]
figa, figb, figc = [scatter(
    mad_res[:,1], 
    mad_res[:,k_list[k]], 
    yerr=mad_res[:,[k_list[k]+3,k_list[k]+6]], 
    c=:black,
    markerstrokecolor=:black,
    xlabel=L"\sigma", 
    ylabel=ylabels[k], 
    legend=false) for k ∈ 1:3]
fig3 = plot(figa, figb, figc, layout=@layout [a;b;c])
savefig(fig3, "Figure3.pdf")


####################################################################################
#   FIGURE 4: SIMULATE GROUPS AND PARAMETER RECOVERY
####################################################################################
println("PLOTTING FIGURE 4: SIMULATED GROUP STUDIES AND PARAMETER RECOVERY")
sample_group(nsubjects, g, a, b, sdev, trange) = @pipe [
        sample_trajectory(g, a, b, sdev, trange)[2] for i ∈ 1:nsubjects
    ] |> mapreduce(permutedims, vcat, _)    
    
function simulate_study(n, dg, da, db; g0=-0.5, a0=5, b0=1, sd=0.1, trange=0:1:21)
    X = [ 
        sample_group(n, g0, a0, b0, sd, trange) ; 
        sample_group(n, g0+dg, a0+da, b0+db, sd, trange) 
    ]
    Y = [zeros(n); ones(n)]

    return (trange |> collect,X,Y)
end

function make_study_df(times, X, Y)
    n = length(Y)
    df = DataFrame() 
    for i ∈ 1:n 
        n_steps = length(times)
        df = [df; DataFrame(
            Subject = categorical(i*ones(n_steps) .|> Int), 
            Group = categorical(Y[i]*ones(n_steps) .|> Int),
            Time = times, 
            Rating = X[i,:]
        )]
    end
    return df 
end

function fit_data(t, y; maxiters=5)
    done = false; n_iters = 0; out = nothing
    while !done
        n_iters += 1
        function loss(p)
            if (p[3] > p[2]) | (p[1] < -1) | (p[1] > 1)
                out = 1e3
            else 
                out = sum((y .- Gnorm(t, p)).^2)
            end 
            return out
        end
        
        res = optimize(loss, [-1., 0.01, 0.01], [1., Inf, Inf], [0., 2., 1.])
        
        if (n_iters >= maxiters) | Optim.converged(res)
            done = true 
            out = res
        end 
    end
    return out
end

function simulate_and_fit(n, dg, da, db, experiment_id; run=1, g0=-0.5, a0=5, b0=1, sd=0.1, trange=0:1:21)
    study = @pipe simulate_study(n, dg, da, db; g0=g0, a0=a0, b0=b0, sd=sd, trange=trange) |> 
        make_study_df(_...)
    fm = @formula(Rating ~ Group*Time + (1|Subject))
    model = fit(MixedModel, fm, study)

    out = vcat([@pipe fit_data(study[study.Subject .== i,:Time], study[study.Subject .== i, :Rating]) |> 
        Optim.minimizer |> 
        DataFrame(
            Subject=categorical([i]), 
            Group=categorical([study[study.Subject .== i, :Group][1]]),
            g=_[1], 
            a=_[2], 
            b=_[3],
            tpeak=- ((_[2]*_[3])/(_[2]-_[3]))*log(_[3]/_[2]))
        for i ∈ 1:length(unique(study.Subject))]...)

    out_summ= combine(groupby(out, [:Group]), 
        :g => mean, :g => std,
        :a => mean, :a => std,
        :b => mean, :b => std,
        :tpeak => mean, :tpeak => std)

    b_fm = @formula(b ~ Group + (1|Subject))
    b_model = fit(MixedModel, b_fm, out)
    
    a_fm = @formula(a ~ Group + (1|Subject))
    a_model = fit(MixedModel, a_fm, out)
    
    g_fm = @formula(g ~ Group + (1|Subject))
    g_model = fit(MixedModel, g_fm, out)

    out_df = DataFrame(
        Experiment=experiment_id,
        Run=run,
        N = 2*n, 
        g0=g0,
        a0=a0,
        b0=b0,
        tpeak0= ftpeak(a0,b0),
        g=g0+dg,
        a=a0+da,
        b=b0+db,
        tpeak= ftpeak(a0+da,b0+db),
        mm_group_p=model.pvalues[2],
        mm_time_p=model.pvalues[3],
        mm_group_by_time_p=model.pvalues[4],
        mm_group_sig=Int(model.pvalues[2] < 0.05),
        mm_time_sig=Int(model.pvalues[3] < 0.05),
        mm_group_by_time_sig=Int(model.pvalues[4] < 0.05),
        g0_mean=out_summ[out_summ.Group .== 0,:].g_mean,
        g0_std=out_summ[out_summ.Group .== 0,:].g_std,
        a0_mean=out_summ[out_summ.Group .== 0,:].a_mean,
        a0_std=out_summ[out_summ.Group .== 0,:].a_std,
        b0_mean=out_summ[out_summ.Group .== 0,:].b_mean,
        b0_std=out_summ[out_summ.Group .== 0,:].b_std,
        tpeak0_mean=out_summ[out_summ.Group .== 0,:].tpeak_mean,
        tpeak0_std=out_summ[out_summ.Group .== 0,:].tpeak_std,
        g_mean=out_summ[out_summ.Group .== 1,:].g_mean,
        g_std=out_summ[out_summ.Group .== 1,:].g_std,
        a_mean=out_summ[out_summ.Group .== 1,:].a_mean,
        a_std=out_summ[out_summ.Group .== 1,:].a_std,
        b_mean=out_summ[out_summ.Group .== 1,:].b_mean,
        b_std=out_summ[out_summ.Group .== 1,:].b_std,
        tpeak_mean=out_summ[out_summ.Group .== 1,:].tpeak_mean,
        tpeak_std=out_summ[out_summ.Group .== 1,:].tpeak_std,
        g_p = g_model.pvalues[2],
        a_p = a_model.pvalues[2],
        b_p = b_model.pvalues[2],
        g_sig = Int(g_model.pvalues[2] < 0.05),
        a_sig = Int(a_model.pvalues[2] < 0.05),
        b_sig = Int(b_model.pvalues[2] < 0.05)
    )

    return out_df
end

function plot_summary_curves(df)
    fig = plot(xlabel=L"t", ylabel=L"\Delta")
    ls = [:dash, :solid]
    colors = [:gray, :black]
    labels = ["Control", "Intervention"]
    for i ∈ 0:1
        fig = @pipe df |>  
            combine(groupby(_, [:Group, :Time]), :Rating => mean, :Rating => std) |> 
            _[_.Group .== i,:] |> 
            plot!(_.Time, _.Rating_mean, yerr=_.Rating_std, c=colors[i+1], ls=ls[i+1], label=labels[i+1])
    end 
    return fig 
end


### Simulate 
df_list = []
for i in 1:Threads.nthreads()
    push!(df_list, [])
end

# Create list of parameters to be simulated
param_list = vcat(
    [[20, 0, 0, db, 1] for db ∈ 0.25:0.25:3], 
    [[20, -dg, 0, 0, 2] for dg ∈ 0.01:0.02:0.2],
    [[20, 0, da, 0, 3] for da ∈ 0.25:0.25:3])

@threads for θ ∈ param_list
    for i ∈ 1:10
        push!(
            df_list[Threads.threadid()],
            simulate_and_fit(Int(θ[1]), θ[2], θ[3], θ[4], θ[5]; run=i, trange=0:1:21)
        )
    end 
end

# SAVE RESULTS 
save("df_list.jld", "df_list", df_list)
out_df = vcat([vcat(df_list[i]...) for i ∈ 1:Threads.nthreads()]...)
save("out_df.jld", "out_df", out_df)
CSV.write("groupstudy-sim.csv", out_df)

# NOW PLOT FIGURE 4
# Load data and compute the perturbation magnitudes
data = CSV.read("groupstudy-sim.csv", DataFrame)
data.dg = abs.(data.g .- data.g0)
data.da = abs.(data.a .- data.a0)
data.db = abs.(data.b .- data.b0)
data.db_est = data.b_mean .- data.b0_mean
data.da_est = data.a_mean .- data.a0_mean

# Define function for standard error computation
serr(x) = std(x)/sqrt(length(x))

# Plotting function
function plot_power_res(i, j)
    titles = [L"\hat{b}", L"\hat{g}", L"\hat{a}"]
    xlabels = [L"\delta b", L"\delta g", L"\delta a"]
    xvars = [:db, :dg, :da]; xvar = xvars[i]
    series_mean = [:b_sig_mean, :g_sig_mean, :a_sig_mean]
    series_se = [:b_sig_serr, :g_sig_serr, :a_sig_serr]
    fontsize = 8
    if i == 2 && j == 2
        fig = plot(xlabel=xlabels[i], ylabel="Power", title=titles[j], 
            legend=:bottomright,
            xtickfont=font(12), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2), 
            xrotation=45)
    elseif i == 2
        fig = plot(xlabel=xlabels[i], ylabel="Power", legend=false, title=titles[j], 
            xtickfont=font(12), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2), 
            xrotation=45)
    else 
        fig = plot(xlabel=xlabels[i], ylabel="Power", legend=false, title=titles[j], 
            xtickfont=font(12), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2))
    end
    df = combine(groupby(data[data.Experiment .== i,:], xvars), 
        [ 
            var => stat 
            for var ∈ [
                :mm_group_sig, :mm_time_sig, :mm_group_by_time_sig, 
                :g_sig, :a_sig, :b_sig]
            for stat ∈ [mean, serr]
        ]...)
    sort!(df, [xvar])
    fig = @pipe (df[:, xvar] .> 0) |> 
        plot!(df[_, xvar], df[_, series_mean[j]], 
            yerr=df[_, series_se[j]], 
            markers=:circle, c=:blue, 
            ls=:solid, 
            label="Ours", 
            titlefont=font(fontsize+3),
            xtickfont=font(fontsize), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2))
    fig = @pipe (df[:, xvar] .> 0) |> 
        plot!(df[_, xvar], df.mm_group_sig_mean[_], 
            yerr=df.mm_group_sig_serr[_], 
            markers=:diamond, c=:red, 
            ls=:dash, 
            label="MM-G",
            titlefont=font(fontsize+3),
            xtickfont=font(fontsize), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2))
    fig = @pipe (df[:, xvar] .> 0) |> 
        plot!(df[_, xvar], df.mm_group_by_time_sig_mean[_], 
            yerr=df.mm_group_by_time_sig_serr[_],
            markers=:square, c=:green, 
            ls=:dot, 
            label="MM-G×T", 
            titlefont=font(fontsize+3),
            xtickfont=font(fontsize), 
            ytickfont=font(fontsize), 
            guidefont=font(fontsize), 
            legendfont=font(fontsize-2))
    return fig 
end


figs = [plot_power_res(i,j) for i in 1:3 for j ∈ 1:3]
fig = plot(figs..., layout = @layout [a b c; d e f; g h i])
savefig(fig, "Figure4.pdf")