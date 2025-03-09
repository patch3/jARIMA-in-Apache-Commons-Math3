package arima.api;

import arima.api.models.ArimaModel;
import arima.api.models.ArimaParameterModel;
import org.junit.Test;

import static org.hibernate.validator.internal.util.Contracts.assertNotNull;

public class ArimaModelTest {
    @Test
    public void testForecast() {
        ArimaParameterModel params = new ArimaParameterModel(1, 0, 1, 0, 0, 0, 1);
        double[] data = {100.0, 101.0, 102.0};
        ArimaModel model = new ArimaModel(params, data, 2);
        assertNotNull(model.forecast(1));
    }
}