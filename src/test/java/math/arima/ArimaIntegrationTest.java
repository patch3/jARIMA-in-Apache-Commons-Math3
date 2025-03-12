package math.arima;

import lombok.val;
import math.arima.analytics.Arima;
import math.arima.analytics.Integrator;
import math.arima.models.ForecastResultModel;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ArimaIntegrationTest {
    @Test
    void testArimaForecastWithKnownData() {
        val data = new double[]{10.1, 20.3, 30.5, 40.7, 50.9, 61.1, 71.3, 81.5, 91.7, 101.9, 112.1, 122.3};
        val forecastSize = 3;

        ForecastResultModel result = Arima.forecast_arima(data, forecastSize);

        assertThat(result.getForecast()).hasSize(forecastSize);
        assertThat(result.getUpperBound()).hasSameSizeAs(result.getForecast());
        assertThat(result.getLowerBound()).hasSameSizeAs(result.getForecast());
    }

    @Test
    void testSeasonalArima() {
        // Seasonal data with a clear periodicity (m=12)
        val seasonalData = new double[]{
                100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, // Season 1
                340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560  // Season 2
        };
        ForecastResultModel result = Arima.forecast_arima(seasonalData, 2);
        assertThat(result.getForecast()).isNotEmpty();
    }

    @Test
    void testDifferentiation() {
        val data = new double[]{5, 10, 15, 20};
        val result = new double[3];
        val initial = new double[]{5};

        Integrator.differentiate(data, result, initial, 1);
        assertThat(result).containsExactly(5.0, 5.0, 5.0);
    }

    @Test
    void testIntegration() {
        val data = new double[]{5, 5, 5};
        val result = new double[4];
        val initial = new double[]{5};

        Integrator.integrate(data, result, initial, 1);
        assertThat(result).containsExactly(5.0, 10.0, 15.0, 20.0);
    }
}