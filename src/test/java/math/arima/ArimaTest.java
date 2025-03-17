package math.arima;

import lombok.val;
import math.series.time.arima.analytics.Arima;
import math.series.time.arima.core.ArimaException;
import math.series.time.arima.models.ArimaForecast;
import math.series.time.arima.models.ArimaModel;
import math.series.time.arima.models.ArimaParameterModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;

class ArimaTest {
    private static final double[] TEST_DATA = {
            10.1, 11.3, 12.5, 13.7, 14.9, 16.1, 17.3, 18.5, 19.7, 20.9
    };

    public static int callDetermineOptimalD(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("determineOptimalD", double[].class);
        method.setAccessible(true);
        return (int) method.invoke(null, (Object) data);
    }

    public static double[] callMakeStationary(double[] data, int d) throws Exception {
        Method method = Arima.class.getDeclaredMethod("makeStationary", double[].class, int.class);
        method.setAccessible(true);
        return (double[]) method.invoke(null, data, d);
    }

    // A test of the basic forecasting functionality
    @Test
    void testBasicForecast() {
        val result = Arima.forecast(TEST_DATA, 3);

        assertNotNull(result, "The forecast result should not be null");
        assertEquals(3, result.getForecast().length, "The length of the forecast must match the requested one");
        assertNotEquals(0.0, result.getForecast()[0], "The predicted values should not be zero");
        assertTrue(result.getRmse() >= 0, "RMSE cannot be negative");
    }

    @Test
    void testForecast() throws ArimaException {
        // Generating data with increased amplitude to avoid numerical errors
        double[] data = generateStationaryData(100, 0.5);
        Arima arima = new Arima(data);
        int forecastSize = 5;
        ArimaForecast forecast = arima.forecast(forecastSize);

        assertNotNull(forecast, "Прогноз не должен быть null");
        assertEquals(forecastSize, forecast.getForecast().length, "Неверная длина прогноза");

        // Checking for the meaningfulness of the AIC (maybe negative in a correct model)
        assertNotEquals(-1.0, forecast.getAic(),
                "AIC не должен быть значением по умолчанию (-1). Текущее значение: " + forecast.getAic());

        // Checking that RMSE is positive
        assertTrue(forecast.getRmse() >= 0,
                "RMSE должен быть неотрицательным. Текущее значение: " + forecast.getRmse());
    }

    @Test
    void testDetermineOptimalDForStationaryData() throws Exception {
        val stationaryData = new double[]{2.1, 1.9, 2.0, 2.05, 1.95};
        val d = callDetermineOptimalD(stationaryData);
        assertEquals(0, d, "Order d should be 0 for stationary data");
    }

    // Stationary check test
    @Test
    void testIsStationary() throws Exception {
        assertTrue(TestUtils.callIsStationary(new double[]{1.0, 1.1, 0.9, 1.05, 0.95}));
        assertFalse(TestUtils.callIsStationary(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}));
        assertTrue(TestUtils.callIsStationary(new double[]{2.0, 2.0, 2.0, 2.0}));
        assertFalse(TestUtils.callIsStationary(new double[]{2.0, 1.0, 5.0, 9.0, 0.4}));
    }

    // Stationary row creation test
    @Test
    void testMakeStationary() throws Exception {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] stationary = callMakeStationary(data, 1);
        assertTrue(TestUtils.callIsStationary(stationary));
    }

    // AIC Calculation Test
    @Test
    void testCalculateAIC() throws Exception {
        val params = new ArimaParameterModel(1, 1, 1, 0, 0, 0, 12);
        val model = new ArimaModel(params, new double[10], 5);
        double aic = TestUtils.callCalculateModelAIC(model, new double[]{1.0, 2.0, 3.0});
        assertFalse(Double.isNaN(aic), "AIC should be a valid number");
    }

    // Invalid input data processing test
    @Test
    void testInvalidInputHandling() {
        assertThrows(IllegalArgumentException.class, () ->
                        Arima.forecast(new double[0], 5),
                "Empty data should throw exception"
        );
        assertThrows(ArimaException.class, () ->
                        Arima.forecast(TEST_DATA, -1),
                "Negative forecast size should throw exception"
        );
    }

    // Seasonal Patterns Test
    @Test
    void testSeasonalityDetection() throws Exception {
        double[] seasonalData = TEST_DATA;
        assertFalse(TestUtils.callIsStationary(seasonalData));

        int d = callDetermineOptimalD(seasonalData);
        assertTrue(d >= 1, "Seasonal data should require differentiation");
    }

    private double[] generateStationaryData(int size, double noiseAmplitude) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = (Math.random() * 2 - 1) * noiseAmplitude;
        }
        return data;
    }
}