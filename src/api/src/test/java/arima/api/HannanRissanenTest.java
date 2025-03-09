package arima.api;

import arima.api.analytics.HannanRissanen;
import arima.api.models.ArimaParameterModel;
import org.junit.Test;

import static org.hibernate.validator.internal.util.Contracts.assertNotNull;

public class HannanRissanenTest {

    @Test
    public void testARMAEstimation() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        ArimaParameterModel params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);

        // length = 8 - 1 = 7 >= 2*r (2*2=4)
        HannanRissanen.estimateARMA(data, params, 1, 3);

        assertNotNull(params.getCurrentARCoefficients());
    }
}