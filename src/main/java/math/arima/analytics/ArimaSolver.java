package math.arima.analytics;

import lombok.val;
import math.arima.core.ArimaException;
import math.arima.models.ArimaModel;
import math.arima.models.ArimaParameterModel;
import math.arima.models.ForecastResultModel;


/**
 * Main solver for ARIMA. Contains forecasting and quality evaluation methods.
 */
public final class ArimaSolver {
    private static final int maxIterationForHannanRissanen = 5;

    /**
     * Performs forecasting for a stationary ARMA model.
     *
     * @param params          the model parameters
     * @param dataStationary  the stationary data
     * @param startIndex      the start index for forecasting
     * @param endIndex        the end index for forecasting
     * @return an array of forecasted values
     */
    public static double[] forecastARMA(final ArimaParameterModel params, final double[] dataStationary,
                                        final int startIndex, final int endIndex) {
        val errors = new double[endIndex];
        val data = new double[endIndex];
        System.arraycopy(dataStationary, 0, data, 0, startIndex);

        val forecast_len = endIndex - startIndex;
        val forecasts = new double[forecast_len];
        val dp = params.getDegreeP();
        val dq = params.getDegreeQ();
        val start_idx = Math.max(dp, dq);

        for (var j = 0; j < start_idx; ++j) {
            errors[j] = 0;
        }
        // populate errors and forecasts
        for (var j = start_idx; j < startIndex; ++j) {
            val forecast = params.forecastOnePointARMA(data, errors, j);
            val error = data[j] - forecast;
            errors[j] = error;
        }
        // now we can forecast
        for (var j = startIndex; j < endIndex; ++j) {
            val forecast = params.forecastOnePointARMA(data, errors, j);
            data[j] = forecast;
            errors[j] = 0;
            forecasts[j - startIndex] = forecast;
        }
        // return forecasted values
        return forecasts;
    }

    public static ForecastResultModel forecastARIMA(final ArimaParameterModel params, final double[] data,
                                                    final int forecastStartIndex, final int forecastEndIndex) {
        val forecastLength = validateAndGetForecastLength(params, data, forecastStartIndex, forecastEndIndex);
        val forecast = new double[forecastLength];
        val diffResult = prepareDifferentiation(params, data, forecastStartIndex);

        //==========================================

        //==========================================
        // FORECAST
        val forecastStationary = forecastARMA(params, diffResult.dataStationary,
                diffResult.dataStationary.length,
                diffResult.dataStationary.length + forecastLength);

        val dataForecastStationary = new double[diffResult.dataStationary.length + forecastLength];

        System.arraycopy(diffResult.dataStationary, 0, dataForecastStationary, 0, diffResult.dataStationary.length);
        System.arraycopy(forecastStationary, 0, dataForecastStationary, diffResult.dataStationary.length,
                forecastStationary.length);
        // END OF FORECAST
        //==========================================

        //=========== UN-CENTERING =================
        Integrator.shift(dataForecastStationary, diffResult.meanStationary);
        //==========================================

        //===========================================
        // INTEGRATE
        val forecast_merged = integrate(params, dataForecastStationary, diffResult.hasSeasonalI, diffResult.hasNonSeasonalI);
        // END OF INTEGRATE
        //===========================================
        System.arraycopy(forecast_merged, forecastStartIndex, forecast, 0, forecastLength);

        return new ForecastResultModel(forecast, Integrator.computeVariance(diffResult.dataStationary));
    }

    public static ArimaModel estimateARIMA(final ArimaParameterModel params, final double[] data,
                                           final int forecastStartIndex, final int forecastEndIndex) {
        val forecast_length = validateAndGetForecastLength(params, data, forecastStartIndex, forecastEndIndex);
        val diffResult = prepareDifferentiation(params, data, forecastStartIndex);
        val data_stationary = diffResult.dataStationary;
        //==========================================
        // FORECAST
        HannanRissanen.estimateARMA(data_stationary, params, forecast_length, maxIterationForHannanRissanen);
        return new ArimaModel(params, data, forecastStartIndex);
    }

    /**
     * Data validation and calculation of forecast length
     */
    private static int validateAndGetForecastLength(ArimaParameterModel params, double[] data,
                                                    int forecastStartIndex, int forecastEndIndex) {
        if (!checkARIMADataLength(params, data, forecastStartIndex, forecastEndIndex)) {
            val initialConditionSize = params.d + params.D * params.m;
            throw new ArimaException(
                    "not enough data for ARIMA. needed at least " + initialConditionSize +
                            ", have " + data.length + ", startIndex=" + forecastStartIndex +
                            ", endIndex=" + forecastEndIndex
            );
        }
        return forecastEndIndex - forecastStartIndex;
    }

    /**
     * Differentiation and centering of data
     **/
    private static DifferentiationResult prepareDifferentiation(ArimaParameterModel params,
                                                                double[] data,
                                                                int forecastStartIndex) {
        val dataTrain = new double[forecastStartIndex];
        System.arraycopy(data, 0, dataTrain, 0, forecastStartIndex);

        val hasSeasonalI = params.D > 0 && params.m > 0;
        val hasNonSeasonalI = params.d > 0;
        val dataStationary = differentiate(params, dataTrain, hasSeasonalI, hasNonSeasonalI);

        val meanStationary = Integrator.computeMean(dataStationary);
        Integrator.shift(dataStationary, -meanStationary);

        return new DifferentiationResult(dataStationary, meanStationary, hasSeasonalI, hasNonSeasonalI);
    }

    private static double[] differentiate(ArimaParameterModel params, double[] trainingData,
                                          boolean hasSeasonalI, boolean hasNonSeasonalI) {
        double[] dataStationary;  // currently un-centered
        if (hasSeasonalI && hasNonSeasonalI) {
            params.differentiateSeasonal(trainingData);
            params.differentiateNonSeasonal(params.getLastDifferenceSeasonal());
            dataStationary = params.getLastDifferenceNonSeasonal();
        } else if (hasSeasonalI) {
            params.differentiateSeasonal(trainingData);
            dataStationary = params.getLastDifferenceSeasonal();
        } else if (hasNonSeasonalI) {
            params.differentiateNonSeasonal(trainingData);
            dataStationary = params.getLastDifferenceNonSeasonal();
        } else {
            dataStationary = new double[trainingData.length];
            System.arraycopy(trainingData, 0, dataStationary, 0, trainingData.length);
        }
        return dataStationary;
    }

    private static double[] integrate(ArimaParameterModel params, double[] dataForecastStationary,
                                      boolean hasSeasonalI, boolean hasNonSeasonalI) {
        double[] forecast_merged;
        if (hasSeasonalI && hasNonSeasonalI) {
            params.integrateSeasonal(dataForecastStationary);
            params.integrateNonSeasonal(params.getLastIntegrateSeasonal());
            forecast_merged = params.getLastIntegrateNonSeasonal();
        } else if (hasSeasonalI) {
            params.integrateSeasonal(dataForecastStationary);
            forecast_merged = params.getLastIntegrateSeasonal();
        } else if (hasNonSeasonalI) {
            params.integrateNonSeasonal(dataForecastStationary);
            forecast_merged = params.getLastIntegrateNonSeasonal();
        } else {
            forecast_merged = new double[dataForecastStationary.length];
            System.arraycopy(dataForecastStationary, 0, forecast_merged, 0,
                    dataForecastStationary.length);
        }
        return forecast_merged;
    }

    /**
     * Computes the Root Mean Squared Error (RMSE) between two arrays.
     *
     * @param left           the reference values
     * @param right          the forecasted values
     * @param leftIndexOffset the offset in the reference array
     * @param startIndex     the start index for comparison
     * @param endIndex       the end index for comparison
     * @return the RMSE value
     */
    public static double computeRMSE(final double[] left, final double[] right,
                                     final int leftIndexOffset,
                                     final int startIndex, final int endIndex) {
        var square_sum = 0.0;
        for (var i = startIndex; i < endIndex; ++i) {
            val error = left[i + leftIndexOffset] - right[i];
            square_sum += error * error;
        }
        return Math.sqrt(square_sum / (endIndex - startIndex));
    }

    public static double computeAIC(final double[] left, final double[] right,
                                    final int leftIndexOffset,
                                    final int startIndex, final int endIndex) {
        var error_sum = 0.0;
        for (var i = startIndex; i < endIndex; ++i) {
            val error = left[i + leftIndexOffset] - right[i];
            error_sum += Math.abs(error);
        }
        if (error_sum == 0.0) {
            return 0;
        } else {
            return (endIndex - startIndex) * Math.log(error_sum) + 2;
        }
    }

    public static double computeRMSEValidation(final double[] data,
                                               final double testDataPercentage, ArimaParameterModel params) {
        val testDataLength = (int) (data.length * testDataPercentage);
        val trainingDataEndIndex = data.length - testDataLength;

        val arimaModel = estimateARIMA(params, data, trainingDataEndIndex, data.length);

        val forecast = arimaModel.forecast(testDataLength).getForecast();
        return computeRMSE(data, forecast, trainingDataEndIndex, 0, forecast.length);
    }

    public static double computeAICValidation(final double[] data,
                                              final double testDataPercentage, ArimaParameterModel params) {
        val testDataLength = (int) (data.length * testDataPercentage);
        val trainingDataEndIndex = data.length - testDataLength;

        val arimaModel = estimateARIMA(params, data, trainingDataEndIndex,
                data.length);
        val forecast = arimaModel.forecast(testDataLength).getForecast();
        return computeAIC(data, forecast, trainingDataEndIndex, 0, forecast.length);
    }

    public static double setSigma2AndPredicationInterval(final ArimaParameterModel params,
                                                         final ForecastResultModel forecastResult, final int forecastSize) {
        val coeffs_AR = params.getCurrentARCoefficients();
        val coeffs_MA = params.getCurrentMACoefficients();
        return forecastResult
                .setConfInterval(ForecastUtil.confidence_constant_95pct,
                        ForecastUtil.getCumulativeSumOfCoeff(
                                ForecastUtil.ARMAtoMA(coeffs_AR, coeffs_MA, forecastSize)));
    }

    private static boolean checkARIMADataLength(ArimaParameterModel params, double[] data, int startIndex,
                                                int endIndex) {
        boolean result = true;
        val initialConditionSize = params.d + params.D * params.m;
        if (data.length < initialConditionSize || startIndex < initialConditionSize
                || endIndex <= startIndex) {
            result = false;
        }
        return result;
    }

    private record DifferentiationResult(
            double[] dataStationary,
            double meanStationary,
            boolean hasSeasonalI,
            boolean hasNonSeasonalI
    ) {
    }
}
