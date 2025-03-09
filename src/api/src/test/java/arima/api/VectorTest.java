package arima.api;

import arima.api.analytics.Vector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class VectorTest {
    @Test
    public void testDotProduct() {
        double[] data1 = {1.0, 2.0, 3.0};
        double[] data2 = {4.0, 5.0, 6.0};
        Vector v1 = new Vector(data1, false);
        Vector v2 = new Vector(data2, false);
        assertEquals(32.0, v1.dot(v2), 1e-6);
    }

    @Test
    public void testSize() {
        Vector v = new Vector(5, 0.0);
        assertEquals(5, v.size());
    }
}