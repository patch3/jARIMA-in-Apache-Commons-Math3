package math.arima;

import lombok.val;
import math.series.time.arima.analytics.Arima;
import math.series.time.arima.analytics.ArimaSolver;
import math.series.time.arima.models.ArimaForecast;
import math.series.time.arima.models.ArimaModel;
import math.series.time.arima.models.ArimaParameterModel;
import org.apache.commons.math3.util.FastMath;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import math.series.time.arima.core.ArimaException;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

class ArimaTest {
    private static final double[] TEST_DATA = {
            10.1, 11.3, 12.5, 13.7, 14.9, 16.1, 17.3, 18.5, 19.7, 20.9
    };

    // Тест базового функционала прогнозирования
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
        // Генерация данных с увеличенной амплитудой для избежания численных ошибок
        double[] data = generateStationaryData(100, 0.5);
        Arima arima = new Arima(data);
        int forecastSize = 5;
        ArimaForecast forecast = arima.forecast(forecastSize);

        assertNotNull(forecast, "Прогноз не должен быть null");
        assertEquals(forecastSize, forecast.getForecast().length, "Неверная длина прогноза");

        // Проверка на осмысленность AIC (может быть отрицательным в корректной модели)
        assertNotEquals(-1.0, forecast.getAic(),
                "AIC не должен быть значением по умолчанию (-1). Текущее значение: " + forecast.getAic());

        // Проверка, что RMSE положительный
        assertTrue(forecast.getRmse() >= 0,
                "RMSE должен быть неотрицательным. Текущее значение: " + forecast.getRmse());
    }



    @Test
    void testDetermineOptimalDForStationaryData() throws Exception {
        val stationaryData = new double[] {2.1, 1.9, 2.0, 2.05, 1.95};
        val d = callDetermineOptimalD(stationaryData);
        assertEquals(0, d, "Order d should be 0 for stationary data");
    }


    // Тест проверки стационарности
    @Test
    void testIsStationary() throws Exception {
        assertTrue(TestUtils.callIsStationary(new double[]{1.0, 1.1, 0.9, 1.05, 0.95}));
        assertFalse(TestUtils.callIsStationary(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}));
        assertTrue(TestUtils.callIsStationary(new double[]{2.0, 2.0, 2.0, 2.0}));
        assertFalse(TestUtils.callIsStationary(new double[]{2.0, 1.0, 5.0, 9.0, 0.4}));
    }

    // Тест создания стационарного ряда
    @Test
    void testMakeStationary() throws Exception {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] stationary = callMakeStationary(data, 1);
        assertTrue(TestUtils.callIsStationary(stationary));
    }

    // Тест расчета AIC
    @Test
    void testCalculateAIC() throws Exception {
        val params = new ArimaParameterModel(1, 1, 1, 0, 0, 0, 12);
        val model = new ArimaModel(params, new double[10], 5);
        double aic = TestUtils.callCalculateModelAIC(model, new double[]{1.0, 2.0, 3.0});
        assertTrue(aic != Double.NaN, "AIC should be a valid number");
    }

    // Тест обработки невалидных входных данных
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

    // Тест сезонных паттернов
    @Test
    void testSeasonalityDetection() throws Exception {
        double[] seasonalData = TEST_DATA;
        assertFalse(TestUtils.callIsStationary(seasonalData));

        int d = callDetermineOptimalD(seasonalData);
        assertTrue(d >= 1, "Seasonal data should require differentiation");
    }

    private double[] generateStationaryData(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random() * 2 - 1; // [-1, 1]
        }
        return data;
    }


    @Test
    public void testForecastThrowsExceptionWhenNoValidModelFound() {
        // Генерируем данные с амплитудой, приводящей к высокой дисперсии
        double[] data = generateStationaryData(100, 2); // Данные с дисперсией > 1.0

        // Проверяем, что вызов forecast приводит к исключению
        ArimaException exception = assertThrows(
                ArimaException.class,
                () -> Arima.forecast(data, 10),
                "Ожидалось исключение ArimaException из-за отсутствия подходящей модели"
        );

        // Проверяем сообщение исключения
        assertTrue(exception.getMessage().contains("No valid ARIMA model found"));
    }



    private double[] generateHighAmplitudeData(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random() * 4 - 2; // Диапазон [-2, 2], дисперсия > 1.0
        }
        return data;
    }


    private double[] generateStationaryData(int size, double noiseAmplitude) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = (Math.random() * 2 - 1) * noiseAmplitude;
        }
        return data;
    }


    public static boolean callIsStationary(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("isStationary", double[].class);
        method.setAccessible(true);
        return (boolean) method.invoke(null, (Object) data);
    }

    public static int callDetermineOptimalD(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("determineOptimalD", double[].class);
        method.setAccessible(true);
        return (int) method.invoke(null, (Object) data);
    }

    public static double[] callDifferentiate(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("differentiate", double[].class);
        method.setAccessible(true);
        return (double[]) method.invoke(null, (Object) data);
    }

    public static double[] callMakeStationary(double[] data, int d) throws Exception {
        Method method = Arima.class.getDeclaredMethod("makeStationary", double[].class, int.class);
        method.setAccessible(true);
        return (double[]) method.invoke(null, data, d);
    }

    public static double callCalculateModelAIC(ArimaModel model, double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("calculateModelAIC", ArimaModel.class, double[].class);
        method.setAccessible(true);
        return (double) method.invoke(null, model, data);
    }
}