package math.arima;


import math.arima.analytics.ArimaSolver;
import math.arima.models.ArimaParameterModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ArimaSolverTest {
    @Test
    public void testForecastARMA() {
        ArimaParameterModel params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        double[] data = {10.0, 11.0, 12.0, 13.0, 14.0};
        double[] forecast = ArimaSolver.forecastARMA(params, data, 3, 5);
        assertEquals(2, forecast.length);
    }
}