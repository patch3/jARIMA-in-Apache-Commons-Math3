package math.arima;

import math.arima.analytics.Arima;
import math.arima.core.ArimaException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class ArimaExceptionTest {
    @Test
    void testEmptyDataInput() {
        assertThrows(ArimaException.class, () ->
                Arima.forecast_arima(new double[0], 1)
        );
    }

    @Test
    void testInvalidForecastSize() {
        assertThrows(ArimaException.class, () ->
                Arima.forecast_arima(new double[]{1, 2, 3}, -2)
        );
    }
}