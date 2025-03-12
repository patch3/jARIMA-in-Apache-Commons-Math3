package math.series.time.arima.analytics;

import lombok.val;
import math.series.time.arima.core.ArimaException;
import math.series.time.arima.models.ArimaModel;
import math.series.time.arima.models.ArimaParameterModel;
import math.series.time.arima.models.ForecastResultModel;

public final class Arima {
    private static final int MAX_D = 2; // Максимальный порядок дифференцирования
    private static final int SEASONAL_PERIOD = 12; // Сезонный период (напр. 12 для месячных данных)

    public static ForecastResultModel forecast(final double[] data, final int forecastSize) {
        try {
            ForecastResultModel bestModel = null;
            var bestAIC = Double.MAX_VALUE;

            // Automatic determination of the differentiation order d
            val optimalD = determineOptimalD(data);
            val stationaryData = makeStationary(data, optimalD);

            for (int p = 0; p <= 2; ++p) {
                for (int q = 0; q <= 2; ++q) {
                    for (int P = 0; P <= 1; ++P) {
                        for (int D = 0; D <= 1; ++D) {
                            for (int Q = 0; Q <= 1; ++Q) {
                                val params = new ArimaParameterModel(
                                        p, optimalD, q,
                                        P, D, Q,
                                        SEASONAL_PERIOD
                                );
                                try {
                                    val model = ArimaSolver.estimateARIMA(
                                            params, stationaryData,
                                            stationaryData.length,
                                            stationaryData.length + forecastSize
                                    );

                                    val aic = calculateModelAIC(model, stationaryData);

                                    if (aic < bestAIC) {
                                        bestAIC = aic;
                                        bestModel = model.forecast(forecastSize);
                                        bestModel.setAic(aic);
                                    }
                                } catch (Exception e) {
                                    // Пропускаем невалидные комбинации параметров
                                }
                            }
                        }
                    }
                }
            }
            if (bestModel == null) {
                throw new ArimaException("No valid ARIMA model found");
            }
            return bestModel;
        } catch (final Exception ex) {
            throw new ArimaException("Failed to build ARIMA forecast: " + ex.getMessage(), ex);
        }
    }

    private static int determineOptimalD(double[] data) {
        int d = 0;
        double[] currentData = data.clone();
        while (d < MAX_D && !isStationary(currentData)) {
            currentData = differentiate(currentData);
            d++;
        }
        return d;
    }

    private static boolean isStationary(double[] data) throws IllegalArgumentException {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        // Упрощенная проверка стационарности через дисперсию
        if (data.length < 2) return true;

        val mean = Integrator.computeMean(data);
        var variance = 0.0;
        for (double value : data) {
            val delta = value - mean;
            variance +=  delta * delta;
        }
        variance /= (data.length - 1);

        return variance < 1.0;
    }

    private static double[] differentiate(double[] data) {
        if (data.length <= 1) return data;

        val diff = new double[data.length - 1];
        for (int i = 1; i < data.length; i++) {
            diff[i - 1] = data[i] - data[i - 1];
        }
        return diff;
    }

    private static double[] makeStationary(double[] data, int d) {
        var result = data.clone();
        for (int i = 0; i < d; i++) {
            result = differentiate(result);
        }
        return result;
    }

    private static double calculateModelAIC(ArimaModel model, double[] data) {
        // Классическая формула AIC
        int nParams = model.getParams().getNumParamsP() + model.getParams().getNumParamsQ();
        double[] forecasts = model.forecast(data.length).getForecast();

        double sse = 0.0;
        for (int i = 0; i < data.length; i++) {
            sse += Math.pow(data[i] - forecasts[i], 2);
        }
        return data.length * Math.log(sse/data.length) + 2 * nParams;
    }
}
