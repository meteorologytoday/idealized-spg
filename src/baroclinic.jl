#pkg"add Oceananigans, CairoMakie"
using Oceananigans
using Oceananigans.Units
using Printf

println("Start program...")
Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

grid = RectilinearGrid(size = (48, 48, 8),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       topology = (Bounded, Bounded, Bounded),
)


model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())





"""
    ramp(y, Δy)

Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:
```
            y < -Δy/2 => ramp = 0
    -Δy/2 < y < -Δy/2 => ramp = y / Δy
            y >  Δy/2 => ramp = 1
```
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
M² = 1e-7 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers # width of the region of the front
Δb = Δy * M²       # buoyancy jump associated with the front
ϵb = 1e-2 * Δb     # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()

set!(model, b=bᵢ)

using CairoMakie
set_theme!(Theme(fontsize = 20))

# Build coordinates with units of kilometers
x, y, z = 1e-3 .* nodes(grid, (Center(), Center(), Center()))

b = model.tracers.b

fig, ax, hm = heatmap(view(b, 1, :, :),
                      colormap = :deep,
                      axis = (xlabel = "y [km]",
                              ylabel = "z [km]",
                              title = "b(x=0, y, z, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m s⁻²]")

println("Saving figure...")
save("fig_baroclinig_01.png", fig)


println("Start simulation")
simulation = Simulation(model, Δt=20minutes, stop_time=20days)
conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, max_Δt=20minutes)



wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(100))



u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
B = Average(b, dims=1)
U = Average(u, dims=1)
V = Average(v, dims=1)

filename = "baroclinic_adjustment"
save_fields_interval = 0.5day

slicers = (east = (grid.Nx, :, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2Writer(model, (; b, ζ);
                                                 filename = filename * "_$(side)_slice",
                                                 schedule = TimeInterval(save_fields_interval),
                                                 overwrite_existing = true,
                                                 indices)
end

simulation.output_writers[:zonal] = JLD2Writer(model, (; b=B, u=U, v=V);
                                               filename = filename * "_zonal_average",
                                               schedule = TimeInterval(save_fields_interval),
                                               overwrite_existing = true)



@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)



using CairoMakie
filename = "baroclinic_adjustment"

sides = keys(slicers)

slice_filenames = NamedTuple(side => filename * "_$(side)_slice.jld2" for side in sides)

b_timeserieses = (east   = FieldTimeSeries(slice_filenames.east, "b"),
                  north  = FieldTimeSeries(slice_filenames.north, "b"),
                  top    = FieldTimeSeries(slice_filenames.top, "b"))

B_timeseries = FieldTimeSeries(filename * "_zonal_average.jld2", "b")

times = B_timeseries.times
grid = B_timeseries.grid


xb, yb, zb = nodes(b_timeserieses.east)

xb = xb ./ 1e3 # convert m -> km
yb = yb ./ 1e3 # convert m -> km

Nx, Ny, Nz = size(grid)

x_xz = repeat(x, 1, Nz)
y_xz_north = y[end] * ones(Nx, Nz)
z_xz = repeat(reshape(z, 1, Nz), Nx, 1)

x_yz_east = x[end] * ones(Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(reshape(z, 1, Nz), grid.Ny, 1)

x_xy = x
y_xy = y
z_xy_top = z[end] * ones(grid.Nx, grid.Ny)

fig = Figure(size = (1600, 800))

zonal_slice_displacement = 1.2

ax = Axis3(fig[2, 1],
           aspect=(1, 1, 1/5),
           xlabel = "x (km)",
           ylabel = "y (km)",
           zlabel = "z (m)",
           xlabeloffset = 100,
           ylabeloffset = 100,
           zlabeloffset = 100,
           limits = ((x[1], zonal_slice_displacement * x[end]), (y[1], y[end]), (z[1], z[end])),
           elevation = 0.45,
           azimuth = 6.8,
           xspinesvisible = false,
           zgridvisible = false,
           protrusions = 40,
           perspectiveness = 0.7)

n = length(times)

b_slices = (east   = interior(b_timeserieses.east[n], 1, :, :),
            north  = interior(b_timeserieses.north[n], :, 1, :),
            top    = interior(b_timeserieses.top[n], :, :, 1))

# Zonally-averaged buoyancy
B = interior(B_timeseries[n], 1, :, :)

clims = 1.1 .* extrema(b_timeserieses.top[n][:])

kwargs = (colorrange=clims, colormap=:deep, shading=NoShading)

surface!(ax, x_yz_east, y_yz, z_yz;  color = b_slices.east, kwargs...)
surface!(ax, x_xz, y_xz_north, z_xz; color = b_slices.north, kwargs...)
surface!(ax, x_xy, y_xy, z_xy_top;   color = b_slices.top, kwargs...)

sf = surface!(ax, zonal_slice_displacement .* x_yz_east, y_yz, z_yz; color = B, kwargs...)

contour!(ax, y, z, B; transformation = (:yz, zonal_slice_displacement * x[end]),
         levels = 15, linewidth = 2, color = :black)

Colorbar(fig[2, 2], sf, label = "m s⁻²", height = Relative(0.4), tellheight=false)

title = "Buoyancy at t = " * string(round(times[n] / day, digits=1)) * " days"
fig[1, 1:2] = Label(fig, title; fontsize = 24, tellwidth = false, padding = (0, 0, -120, 0))

rowgap!(fig.layout, 1, Relative(-0.2))
colgap!(fig.layout, 1, Relative(-0.1))

save("baroclinic_adjustment_3d.png", fig)

