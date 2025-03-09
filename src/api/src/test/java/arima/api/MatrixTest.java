package arima.api;

import arima.api.analytics.Matrix;
import arima.api.analytics.Vector;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;


public class MatrixTest {
    @Test
    public void testMatrixVectorMultiplication() {
        // 2x3 matrix (m=2 rows, n=3 columns)
        double[][] data = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };
        Matrix m = new Matrix(data, false);

        // Vector of length 3 (n=3)
        Vector v = new Vector(new double[]{5.0, 6.0, 7.0}, false);
        Vector result = m.timesVector(v);

        // [1*5 + 2*6 + 3*7 = 5+12+21=38]
        // [4*5 + 5*6 + 6*7 = 20+30+42=92]
        assertArrayEquals(new double[]{38.0, 92.0}, result.deepCopy(), 1e-6);
    }
}