# Rapid Antidepressant Response Timecourse Simulation 

Code for the paper: Nunes A, Singh S. _A Computational Model to Characterize the Time-Course of Response to Rapid Antidepressant Therapies_.

## Running Code 

Assuming you have 16 threads. This can be changed. 

``` bash
julia --nthreads 16 main.jl
```

The following packages are required: 
- `Plots`
- `Pipe`
- `DataFrames`
- `MixedModels`
- `LaTeXStrings`
- `StatsBase`
- `CategoricalArrays`
- `CSV`
- `JLD`
- `Distributions`
- `StableRNGs`
- `LambertW`
- `Optim`

 
