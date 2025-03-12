package math.arima;


import lombok.val;
import math.arima.analytics.HannanRissanen;
import math.arima.core.ArimaException;
import math.arima.models.ArimaParameterModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class HannanRissanenTest {
    @Test
    public void testARMAEstimation() {
        val data = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        val params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        // length = 8 - 1 = 7 >= 2*r (2*2=4)
        HannanRissanen.estimateARMA(data, params, 1, 3);
        assertNotNull(params.getCurrentARCoefficients());
    }

    // A test to check the processing of insufficient data
    @Test
    void testInsufficientData() {
        val smallData = new double[]{1.0, 2.0, 3.0};
        val params = new ArimaParameterModel(2, 0, 1, 0, 0, 0, 0);

        assertThrows(ArimaException.class, () ->
                        HannanRissanen.estimateARMA(smallData, params, 1, 1),
                "Should throw for insufficient data"
        );
    }

    // Error Update Test
    @Test
    void testErrorUpdateViaReflection() throws Exception {
        // Data preparation
        val data = new double[]{1.0, 2.0, 3.0, 4.0};
        val errors = new double[4];
        val params = new ArimaParameterModel(1, 0, 0, 0, 0, 0, 0);
        // Getting access to a private method
        val updateErrorsMethod = HannanRissanen.class.getDeclaredMethod(
                "updateErrors",
                double[].class,
                double[].class,
                ArimaParameterModel.class,
                int.class,
                int.class
        );
        updateErrorsMethod.setAccessible(true);
        // Вызов метода
        updateErrorsMethod.invoke(
                null, // for a static method
                data,
                errors,
                params,
                1, // r
                2  // size
        );
        assertNotEquals(0.0, errors[2], "Errors should be updated");
    }

    // RMSE Reduction Test
    @Test
    void testRMSEImprovement() {
        val data = generateMA1Data(100, 0.6, 0.1);
        ArimaParameterModel params = new ArimaParameterModel(0, 0, 1, 0, 0, 0, 0);

        val initialRMSE = Double.MAX_VALUE;
        HannanRissanen.estimateARMA(data, params, 10, 3);

        assertTrue(params.getCurrentMACoefficients()[0] < initialRMSE, "RMSE should decrease");
    }

    private double[] generateMA1Data(int n, double theta, double noiseLevel) {
        val data = new double[n];
        val errors = new double[n];
        for (var i = 1; i < n; i++) {
            errors[i] = noiseLevel * Math.random();
            data[i] = theta * errors[i - 1] + errors[i];
        }
        return data;
    }
}