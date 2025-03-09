package arima.api.models;

import arima.api.analytics.ArimaSolver;
import lombok.Getter;
import lombok.Setter;

/**
 * ARIMA Forecast Result
 */
@Getter
public class ForecastResultModel {
    private final double dataVariance;

    private final double[] forecast;
    @Setter
    private double[] upperBound;
    @Setter
    private double[] lowerBound;

    private double modelAIC;
    private double modelRMSE;

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

        this.modelAIC = -1;
        this.modelRMSE = -1;
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

    public double getAIC() {
        return modelAIC;
    }

    void setAIC(double aic) {
        this.modelAIC = aic;
    }

    /**
     * Getter for Root Mean-Squared Error
     *
     * @return Root Mean-Squared Error
     */
    public double getRMSE() {
        return this.modelRMSE;
    }

    /**
     * Setter for Root Mean-Squared Error
     *
     * @param rmse Root Mean-Squared Error
     */
    void setRMSE(double rmse) {
        this.modelRMSE = rmse;
    }

    /**
     * Compute and set confidence intervals
     *
     * @param constant          confidence interval constant
     * @param cumulativeSumOfMA cumulative sum of MA coefficients
     * @return Max Normalized Variance
     */
    public double setConfInterval(final double constant, final double[] cumulativeSumOfMA) {
        double maxNormalizedVariance = -1.0;
        double bound;
        for (int i = 0; i < forecast.length; i++) {
            bound = constant * modelRMSE * cumulativeSumOfMA[i];
            this.upperBound[i] = this.forecast[i] + bound;
            this.lowerBound[i] = this.forecast[i] - bound;
            final double normalizedVariance = getNormalizedVariance(Math.pow(bound, 2));
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
