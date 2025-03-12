package math.arima;

import lombok.val;
import math.series.time.arima.analytics.Arima;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import math.series.time.arima.core.ArimaException;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

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
    }

    // Тест автоматического определения дифференцирования
    @Test
    void testDetermineOptimalD() throws Exception {
        // Подготовка тестовых данных
        val nonStationaryData = new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

        // Получение доступа к приватному методу через рефлексию
        val method = Arima.class.getDeclaredMethod(
                "determineOptimalD",
                double[].class
        );
        method.setAccessible(true); // Разрешаем доступ

        // Вызов метода
        val d = (int) method.invoke(null, (Object) nonStationaryData);

        // Проверки
        assertTrue(d >= 1 && d <= 2,
                "Порядок дифференцирования должен быть 1 или 2 для нестационарных данных");
    }

    @Test
    void testDetermineOptimalDForStationaryData() throws Exception {
        // Стационарные данные
        val stationaryData = new double[] {2.1, 1.9, 2.0, 2.05, 1.95};

        val method = Arima.class.getDeclaredMethod("determineOptimalD", double[].class);
        method.setAccessible(true);

        val d = (int) method.invoke(null, stationaryData);

        assertEquals(0, d,
                "Для стационарных данных порядок дифференцирования должен быть 0");
    }

    @Test
    void testDetermineOptimalDEdgeCases() {
        assertThrows(Exception.class, () -> {
            Method method = Arima.class.getDeclaredMethod("determineOptimalD", double[].class);
            method.invoke(null, (Object) new double[0]); // Пустой массив
        }, "Должно выбрасываться исключение для пустых данных");
    }

    // Тест обработки ошибок при невалидных входных данных
    @Test
    void testInvalidInputHandling() {
        assertThrows(ArimaException.class, () ->
                        Arima.forecast(new double[0], 5),
            "Пустые данные должны вызывать исключение"
        );
        
        assertThrows(ArimaException.class, () -> 
            Arima.forecast(TEST_DATA, -1),
            "Отрицательный размер прогноза должен вызывать исключение"
        );
    }

    // Тест сезонных компонентов
    @Test
    void testStationarityWithNoise() throws Exception {
        // Стационарные данные с небольшим шумом
        double[] stationaryWithNoise = {1.0, 1.05, 0.95, 1.02, 0.98};
        assertTrue(TestUtils.callIsStationary(stationaryWithNoise), "Данные с шумом должны быть стационарными");
    }

    @Test
    void testStationarityCheckWithHelper() throws Exception {
        double[] stationaryData = {1.0, 1.1, 0.9, 1.05, 0.95};
        assertTrue(TestUtils.callIsStationary(stationaryData), "Данные должны определяться как стационарные");

        double[] nonStationaryData = {1.0, 2.0, 3.0, 4.0, 5.0};
        assertFalse(TestUtils.callIsStationary(nonStationaryData), "Трендовые данные не стационарны");
    }

    @Test
    void testSeasonalNonStationarity() throws Exception {
        Method isStationaryMethod = Arima.class.getDeclaredMethod("isStationary", double[].class);
        isStationaryMethod.setAccessible(true);

        // Тест 5: Сезонные колебания
        double[] seasonalData = {10, 20, 30, 10, 20, 30, 10};
        boolean result = (boolean) isStationaryMethod.invoke(null, seasonalData);
        assertFalse(result, "Сезонные паттерны должны определяться как нестационарные");
    }

    @Test
    void testStationarityCheck() throws Exception {
        // Получаем доступ к приватному методу через рефлексию
        Method isStationaryMethod = Arima.class.getDeclaredMethod(
                "isStationary",
                double[].class
        );
        isStationaryMethod.setAccessible(true); // Разрешаем доступ

        // Тестовые данные
        double[] stationaryData = {1.0, 1.1, 0.9, 1.05, 0.95};
        double[] nonStationaryData = {1.0, 2.0, 3.0, 4.0, 5.0};

        // Вызов метода для стационарных данных
        boolean isStationaryResult = (boolean) isStationaryMethod.invoke(null, stationaryData);
        assertTrue(isStationaryResult, "Данные должны определяться как стационарные");

        // Вызов метода для нестационарных данных
        boolean isNonStationaryResult = (boolean) isStationaryMethod.invoke(null, nonStationaryData);
        assertFalse(isNonStationaryResult, "Трендовые данные не стационарны");
    }

    @Test
    void testStationarityEdgeCases() throws Exception {
        Method isStationaryMethod = Arima.class.getDeclaredMethod("isStationary", double[].class);
        isStationaryMethod.setAccessible(true);

        // Проверка пустого массива
        InvocationTargetException exception = assertThrows(InvocationTargetException.class, () -> {
            isStationaryMethod.invoke(null, new double[0]);
        }, "Должно выбрасываться исключение для пустых данных");

        // Проверяем исходное исключение внутри InvocationTargetException
        assertTrue(exception.getCause() instanceof IllegalArgumentException);
        assertEquals("Input data cannot be null or empty", exception.getCause().getMessage());

        // Проверка массива с одним элементом
        boolean singleElementResult = (boolean) isStationaryMethod.invoke(null, new double[]{1.0});
        assertTrue(singleElementResult);

        // Проверка постоянных значений
        boolean constantDataResult = (boolean) isStationaryMethod.invoke(null, new double[]{1.0, 1.0, 1.0});
        assertTrue(constantDataResult);
    }
}