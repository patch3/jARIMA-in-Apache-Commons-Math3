package math.arima.models;

import lombok.Getter;
import lombok.Setter;
import lombok.val;
import math.arima.analytics.ArimaSolver;

/**
 * ARIMA model
 */
@Getter
public class ArimaModel {
    private final ArimaParameterModel params;
    private final double[] data;
    private final int trainDataSize;
    @Setter
    private double rmse;
    @Setter
    private double aic;

    /**
     * Constructor for ArimaModel
     *
     * @param params        ARIMA parameter
     * @param data          original data
     * @param trainDataSize size of train data
     */
    public ArimaModel(ArimaParameterModel params, double[] data, int trainDataSize) {
        this.params = params;
        this.data = data;
        this.trainDataSize = trainDataSize;
    }

    /**
     * Forecast data base on training data and forecast size.
     *
     * @param forecastSize size of forecast
     * @return forecast result
     */
    public ForecastResultModel forecast(final int forecastSize) {
        val forecastResult = ArimaSolver.forecastARIMA(
                params, data, trainDataSize, trainDataSize + forecastSize
        );
        forecastResult.setAic(this.aic);
        forecastResult.setRmse(this.rmse);
        return forecastResult;
    }
}
