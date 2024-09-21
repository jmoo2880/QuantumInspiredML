using JLD2
using Plots


f = jldopen("Data/NASA_kepler/datasets/KeplerLightCurveOrig.jld2", "r");
X_train = read(f, "X_train");
y_train = read(f, "y_train");
X_test = read(f, "X_test");
y_test = read(f, "y_test");

function class_distribution()

end