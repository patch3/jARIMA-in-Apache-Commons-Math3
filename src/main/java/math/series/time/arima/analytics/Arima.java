package math.series.time.arima.analytics;

import lombok.val;
import math.series.time.TimeSeries;
import math.series.time.arima.core.ArimaException;
import math.series.time.arima.models.ArimaModel;
import math.series.time.arima.models.ArimaParameterModel;
import math.series.time.arima.models.ArimaForecast;
import org.apache.commons.math3.util.FastMath;

public final class Arima extends TimeSeries<ArimaForecast> {
    private static final int MAX_D = 2; // Максимальный порядок дифференцирования
    private static final int SEASONAL_PERIOD = 12; // Сезонный период (напр. 12 для месячных данных)

    public Arima(double[] data) {
        super(data);
    }

    @Override
    public ArimaForecast forecast(int forecastSize) {
        return forecast(data, forecastSize);
    }

    public static ArimaForecast forecast(final double[] data, final int forecastSize) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        try {
            ArimaForecast bestModel = null;
            var bestAIC = Double.MAX_VALUE;

            // Automatic determination of the differentiation order d
            val optimalD = determineOptimalD(data);
            val stationaryData = makeStationary(data, optimalD);

            for (int p = 0; p <= 3; ++p) {
                for (int q = 0; q <= 3; ++q) {
                    for (int P = 0; P <= 2; ++P) {
                        for (int D = 0; D <= 2; ++D) {
                            for (int Q = 0; Q <= 2; ++Q) {
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
                                    e.printStackTrace();
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
        var d = 0;
        var currentData = data.clone();
        while (d < MAX_D && !isStationary(currentData)) {
            currentData = differentiate(currentData);
            ++d;
        }
        return d;
    }

    private static boolean isStationary(double[] data) throws IllegalArgumentException {
        val mean = Integrator.computeMean(data);
        var variance = 0.0;
        for (double value : data) {
            val delta = value - mean;
            variance +=  delta * delta;
        }
        variance /= (data.length - 1);

        return variance < 1.0;
    }

    private static double[] differentiate(double[] data) throws ArimaException {
        if (data.length <= 1) return data;

        val diff = new double[data.length - 1];
        double[] initial = {data[0]}; // начальное условие для дифференцирования

        // Используем метод из Integrator
        Integrator.differentiate(data, diff, initial, 1);
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
        val nParams = model.getParams().getNumParamsP() + model.getParams().getNumParamsQ();
        val forecasts = model.forecast(data.length).getForecast();
        var sse = 0.0;
        for (int i = 0; i < data.length; i++) {
            var error = data[i] - forecasts[i];
            sse += error * error;
        }
        return data.length * FastMath.log(sse / data.length) + 2 * nParams;
    }
}
