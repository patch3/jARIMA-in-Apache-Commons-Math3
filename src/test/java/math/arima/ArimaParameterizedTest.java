package math.arima;

import lombok.val;
import math.arima.analytics.Arima;
import math.arima.models.ForecastResultModel;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;

public class ArimaParameterizedTest {
    @ParameterizedTest
    @CsvSource({
            "0, 1, 0",  // Simple model
            "1, 0, 1",  // ARMA(1,1)
            "2, 1, 2"   // ARIMA(2,1,2)
    })
    void testDifferentParameters(int p, int d, int q) {
        val data = new double[]{10.1, 20.3, 30.5, 40.7, 50.9, 61.1, 71.3, 81.5, 91.7, 101.9, 112.1, 122.3};
        ForecastResultModel result = Arima.forecast_arima(data, 2);
        assertThat(result.getForecast()).isNotNull();
    }
}