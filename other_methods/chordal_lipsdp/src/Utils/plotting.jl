
# Generate trajectories from a unit box
function randomTrajectories(N::Int, ffnet::NeuralNetwork, x1min::VecF64, x1max::VecF64)
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]
  xgaps = x1max - x1min
  box01_points = rand(ffnet.xdims[1], N)
  x1s = [x1min + (p .* xgaps) for p in eachcol(box01_points)]
  xfs = [runNetwork(x1, ffnet) for x1 in x1s]
  return xfs
end

# Plot some data to a file
function plotRandomTrajectories(N::Int, ffnet::NeuralNetwork;
                                x1min = -ones(ffnet.xdims[1]),
                                x1max = ones(ffnet.xdims[1]),
                                saveto = "~/Desktop/hello.png")
  # Make sure we can actually plot these in 2D
  @assert ffnet.xdims[end] == 2
  xfs = randomTrajectories(N, ffnet, x1min, x1max)
  d1s = [xf[1] for xf in xfs]
  d2s = [xf[2] for xf in xfs]
  p = scatter(d1s, d2s, markersize=2, alpha=0.3)
  savefig(p, saveto)
end

# Plot different line data
# Get data of form (label1, ys1), (label2, ys2), ...
function plotLines(xs, labeled_lines::Vector{Tuple{String, VecF64}};
                   title = "title", ylogscale = false, saveto = "~/Desktop/foo.png")
  # Make sure we have a consistent number of data
  @assert all(lys -> length(xs) == length(lys[2]), labeled_lines)
  plt = plot(title=title)
  colors = theme_palette(:auto)
  for (i, (lbl, ys)) in enumerate(labeled_lines)
    if ylogscale
      plot!(xs, ys, label=lbl, color=colors[i], yscale=:log10)
    else
      plot!(xs, ys, label=lbl, color=colors[i])
    end
  end
  savefig(plt, saveto)
  return plt
end

