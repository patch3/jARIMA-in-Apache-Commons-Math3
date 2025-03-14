package math.series.time;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public abstract class ForecastResult {
    protected double[] forecast;

    public ForecastResult(double[] forecast) {
        this.forecast = forecast;
    }
}
