package math.arima;


import math.arima.analytics.ArimaSolver;
import math.arima.models.ArimaParameterModel;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ArimaSolverTest {
    @Test
    public void testForecastARMA() {
        ArimaParameterModel params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        double[] data = {10.0, 11.0, 12.0, 13.0, 14.0};
        double[] forecast = ArimaSolver.forecastARMA(params, data, 3, 5);
        assertEquals(2, forecast.length);
    }

    @Test
    void testRMSEComputation() {
        double[] actual = {10, 20, 30};
        double[] forecast = {12, 18, 28};
        double rmse = ArimaSolver.computeRMSE(actual, forecast, 0, 0, 3);
        assertThat(rmse).isBetween(2.0, 3.0);
    }

    @Test
    void testAICComputation() {
        ArimaParameterModel params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double aic = ArimaSolver.computeAICValidation(data, 0.2, params);
        assertThat(aic).isNotNegative();
    }
}