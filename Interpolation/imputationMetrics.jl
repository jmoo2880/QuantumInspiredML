using StatsBase
using PrettyTables

function mape(forecast::Vector{Float64}, actual::Vector{Float64}; symmetric=false)
    """
    Compute (symmetric) Mean Absolute Percentage Error 
    (sMAPE) MPAE given time series forecast window and ground-truth. 
    """

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

########################### scale dependent errors ########################
function mse(forecast::Vector{Float64}, actual::Vector{Float64})
    """Mean squared error"""
    @assert isequal(length(forecast), length(actual)) "Forecast and ground truth time series do not match in length."
    sq_error = (abs.(forecast - actual)).^2
    mean_sq_error = mean(sq_error)

    return mean_sq_error

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

function rmse(forecast::Vector{Float64}, actual::Vector{Float64})
    """ Root mean squared error (RMSE)."""

    return sqrt(mse(forecast, actual))

end

#################################################################

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

# excluded MASE due to needing to set a seasonal period - depends on dataset. 
function compute_all_forecast_metrics(forecast::Vector{Float64}, 
    actual::Vector{Float64}, print_table::Bool=true)
    """Compute all metrics for a single forecast.
    Will compute the following: 
    - MAPE
    - sMAPE
    - MAE
    - MSE
    - RMSE
    Forecast and actual correspond to the forecasted time pts. and actual time pts. (not including training data)
    training data is only the time pts. used for generating the predictions.
    """
    metric_outputs = Dict(
        :MAPE => mape(forecast, actual),
        :SMAPE => mape(forecast, actual; symmetric=true),
        :MAE => mae(forecast, actual),
        :MSE => mse(forecast, actual),
        :RMSE => mse(forecast, actual),
    );
    if print_table
        pretty_table(metric_outputs; header=["Metric", "Value"], header_crayon= crayon"yellow bold", tf = tf_unicode_rounded);
    end

    return metric_outputs

end

