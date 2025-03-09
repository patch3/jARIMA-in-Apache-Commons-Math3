package arima.api.models;

import lombok.Data;

@Data
public class TimeSeriesModel {
    private final double[] tsData;
    private final int forecastPeriod;

    /**
     * Constructor for TimeSeriesModel
     */
    public TimeSeriesModel(double[] tsData, int forecastPeriod) {
        this.forecastPeriod = forecastPeriod;
        this.tsData = tsData;
    }
}
