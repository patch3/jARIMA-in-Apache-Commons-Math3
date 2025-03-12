package math.arima.models;

import lombok.Getter;
import lombok.Setter;
import lombok.val;
import math.arima.analytics.ArimaSolver;

/**
 * ARIMA forecast result. Contains point forecasts and confidence intervals.
 */
@Getter
public class ForecastResultModel {
    private final double dataVariance;

    private final double[] forecast;
    @Setter
    private double[] upperBound;
    @Setter
    private double[] lowerBound;
    @Setter
    private double aic;
    @Setter
    private double rmse;

    private double maxNormalizedVariance;

    /**
     * Constructor for ForecastResult
     */
    public ForecastResultModel(final double[] pForecast, final double pDataVariance) {
        this.forecast = pForecast;

        this.upperBound = new double[pForecast.length];
        System.arraycopy(pForecast, 0, upperBound, 0, pForecast.length);

        this.lowerBound = new double[pForecast.length];
        System.arraycopy(pForecast, 0, lowerBound, 0, pForecast.length);

        this.dataVariance = pDataVariance;

        this.aic = -1;
        this.rmse = -1;
        this.maxNormalizedVariance = -1;
    }

    /**
     * Compute normalized variance
     */
    private double getNormalizedVariance(final double v) {
        if (v < -0.5 || dataVariance < -0.5) {
            return -1;
        } else if (dataVariance < 0.0000001) {
            return v;
        } else {
            return Math.abs(v / dataVariance);
        }
    }


    /**
     * Updates the confidence intervals based on the given confidence level.
     *
     * @param constant          the constant for the confidence level (e.g., 1.96 for 95%)
     * @param cumulativeSumOfMA the cumulative sums of MA coefficients
     * @return the maximum normalized variance
     */
    public double setConfInterval(final double constant, final double[] cumulativeSumOfMA) {
        double maxNormalizedVariance = -1.0;
        double bound;
        for (int i = 0; i < forecast.length; i++) {
            bound = constant * rmse * cumulativeSumOfMA[i];
            this.upperBound[i] = this.forecast[i] + bound;
            this.lowerBound[i] = this.forecast[i] - bound;
            val normalizedVariance = getNormalizedVariance(Math.pow(bound, 2));
            if (normalizedVariance > maxNormalizedVariance) {
                maxNormalizedVariance = normalizedVariance;
            }
        }
        return maxNormalizedVariance;
    }

    /**
     * Compute and set Sigma2 and prediction confidence interval.
     *
     * @param params ARIMA parameters from the model
     */
    public void setSigma2AndPredicationInterval(ArimaParameterModel params) {
        maxNormalizedVariance = ArimaSolver
                .setSigma2AndPredicationInterval(params, this, forecast.length);
    }
}
