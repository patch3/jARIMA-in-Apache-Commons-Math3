package arima.api.models;


import lombok.Data;

/**
 * R ARIMA Forecast Result
 */
@Data
public class RArimaResultModel {
    private final double[] forecast;
    private final double[] upperBound;
    private final double[] lowerBound;

    /**
     * Constructor for RArimaResultModel
     */
    public RArimaResultModel(final double[] forecast,
                             final double[] upperBound, final double[] lowerBound) {

        this.forecast = forecast;
        this.upperBound = upperBound;
        this.lowerBound = lowerBound;
    }
}
