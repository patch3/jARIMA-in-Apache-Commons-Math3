package math.arima;


import lombok.val;
import math.arima.analytics.ArimaSolver;
import math.arima.models.ArimaParameterModel;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ArimaSolverTest {
    @Test
    public void testForecastARMA() {
        val params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        val data = new double[]{10.0, 11.0, 12.0, 13.0, 14.0};
        val forecast = ArimaSolver.forecastARMA(params, data, 3, 5);
        assertEquals(2, forecast.length);
    }

    @Test
    void testRMSEComputation() {
        val actual = new double[]{10, 20, 30};
        val forecast = new double[]{12, 18, 28};
        val rmse = ArimaSolver.computeRMSE(actual, forecast, 0, 0, 3);
        assertThat(rmse).isBetween(2.0, 3.0);
    }

    @Test
    void testAICComputation() {
        val params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        val data = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        val aic = ArimaSolver.computeAICValidation(data, 0.2, params);
        assertThat(aic).isNotNegative();
    }
}