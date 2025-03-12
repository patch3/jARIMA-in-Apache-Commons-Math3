package math.arima;

import math.series.time.arima.analytics.Arima;
import math.series.time.arima.core.ArimaException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class ArimaExceptionTest {
    @Test
    void testEmptyDataInput() {
        assertThrows(ArimaException.class, () ->
                Arima.forecast(new double[0], 1)
        );
    }

    @Test
    void testInvalidForecastSize() {
        assertThrows(ArimaException.class, () ->
                Arima.forecast(new double[]{1, 2, 3}, -2)
        );
    }
}