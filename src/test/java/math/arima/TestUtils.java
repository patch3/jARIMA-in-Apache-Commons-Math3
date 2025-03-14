package math.arima;

import math.series.time.arima.analytics.Arima;
import math.series.time.arima.models.ArimaModel;

import java.lang.reflect.Method;

public class TestUtils {
    public static boolean callIsStationary(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("isStationary", double[].class);
        method.setAccessible(true);
        return (boolean) method.invoke(null, (Object) data);
    }

    public static double callCalculateModelAIC(ArimaModel model, double[] data) throws Exception {
        // Получаем приватный метод
        Method method = Arima.class.getDeclaredMethod(
                "calculateModelAIC",
                ArimaModel.class,
                double[].class
        );
        method.setAccessible(true); // Разрешаем доступ

        // Вызываем статический метод (передаем null как экземпляр)
        return (double) method.invoke(null, model, data);
    }












}