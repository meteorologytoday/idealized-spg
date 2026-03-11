# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %%
using Oceananigans
using Oceananigans.Units
H = 1000meters
W = 20kilometers

grid = RectilinearGrid(CPU(), size=(64, 64, 10), x=(0, W), y=(0, W), z=(-H, 0), topology=(Periodic, Periodic, Bounded))



# %%

# A mountain in the ocean
mountain_h₀ = 250meters
mountain_width = W
hill(x) = mountain_h₀ * exp(-x^2 / 2mountain_width^2)
bottom(x) = - H + hill(x)

grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))


x = xnodes(grid, Center())
bottom_boundary = interior(grid.immersed_boundary.bottom_height, :, 1, 1)
top_boundary = 0 * x




#model = NonhydrostaticModel(; grid, advection=WENO())
#ϵ(x, y) = 2rand() - 1
#set!(model, u=ϵ, v=ϵ)
#simulation = Simulation(model; Δt=0.01, stop_time=4)
#run!(simulation)

# %%
using CairoMakie

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((0, grid.Lx/W), (-grid.Lz, 0)))

band!(ax, x/1e3, bottom_boundary, top_boundary, color = :mediumblue)

fig

