package math.arima;

import math.series.time.arima.analytics.Arima;

import java.lang.reflect.Method;

public class TestUtils {
    public static boolean callIsStationary(double[] data) throws Exception {
        Method method = Arima.class.getDeclaredMethod("isStationary", double[].class);
        method.setAccessible(true);
        return (boolean) method.invoke(null, (Object) data);
    }
}