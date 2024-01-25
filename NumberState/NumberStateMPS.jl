using ITensors
using Random
using StatsBase
using PyCall
pyts = pyimport("pyts.approximation")

function RawTimeSeriesToSAX(time_series::Matrix, n_bins::Int=3, strategy="normal")
    """Function to convert raw time series data to a SAX representation.
    Calls on the SAX library in python using pycall."""
    if strategy !== "normal" || strategy !== "quantile" || strategy !== "uniform"
        error("")
        
    # fit the SAX 'model'
    sax_fit = pyts.SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)




end