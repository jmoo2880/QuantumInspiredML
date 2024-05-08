using StatsBase
using PrettyTables

function mape(forecast::Vector{Float64}, actual::Vector{Float64}; symmetric=false)

    numerator_vals = abs.(actual .- forecast)
    if symmetric
        denominator_vals = (abs.(actual) + abs.(forecast))./2
    else
        denominator_vals = abs.(actual)
    end
    ratios = numerator_vals./denominator_vals
    ratios_sum = sum(ratios)
    average = ratios_sum./(length(forecast))
    
    return average

end

function mae(forecast::Vector{Float64}, actual::Vector{Float64})
    """Mean absolute error (MAE).
    Output is a non-negative float. Best value is 0.0.
    On the same scale as the data and penalises large errors 
    to a lesser degree than MSE or RMSE."""
    @assert isequal(length(forecast), length(actual)) "Forecast and ground truth time series do not match in length."
    error = abs.(forecast - actual)
    mean_abs_error = mean(error)
    return mean_abs_error
end

function mse(forecast::Vector{Float64}, actual::Vector{Float64})
    """Mean squared error"""
    @assert isequal(length(forecast), length(actual)) "Forecast and ground truth time series do not match in length."
    sq_error = (abs.(forecast - actual)).^2
    mean_sq_error = mean(sq_error)
    return mean_sq_error
end

function rmse(forecast::Vector{Float64}, actual::Vector{Float64})
    """ Root mean squared error (RMSE)."""
    return sqrt(mse(forecast, actual))
end

function mase(train::Vector{Float64}, forecast::Vector{Float64}, actual::Vector{Float64}; 
    seasonal_period=1)
    """Mean absolute scaled error (MASE).
    NOTE:
    Computes the MAE of the naive forecast on the training data and the MASE on the test data.
    This means that (actual + train) is the full time series.
    Works for both seasonal and non-seasonal time series.
    """
    error = abs.(forecast - actual)
    mae_forecast = mean(error)
    if seasonal_period == 1
        # one step naive forecast
        mae_naive = mean(abs.(diff(train)))
    else
        # seasonal period ≠ 1
        m = seasonal_period
        diffs_train = [train[i] - train[i-m] for i in (m+1):length(train)]
        mae_naive = mean(abs.(diffs_train))
    end
    return mae_forecast/mae_naive
end

function compute_all_metrics(forecast::Vector{Float64}, 
    actual::Vector{Float64}; print_table=false)
    """Compute all metrics for a single forecast.
    Forecast and actual correspond to the forecasted time pts. and actual time pts. (not including training data)
    training data is only the time pts. used for generating the predictions."""
    metric_outputs = Dict(
        "MAPE" => mape(forecast, actual),
        "sMAPE" => mape(forecast, actual; symmetric=true),
        "MAE" => mae(forecast, actual),
        "MSE" => mse(forecast, actual),
        "RMSE" => mse(forecast, actual),
    )
    if print_table
        pretty_table(metric_outputs)
    end

    return metric_outputs

end
