package math.series.time;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * An abstract base class for time series analysis
 */
@Getter
@AllArgsConstructor
@NoArgsConstructor
public abstract class TimeSeries<T extends ForecastResult> {
    protected double[] data;

    /**
     * Getting historical training data
     */
    public void fit(double[] data) {
        this.data = data.clone();
    }

    /**
     * Forecasting future values
     * @return Forecast result
     */
    public abstract T forecast(int forecastSize);
}