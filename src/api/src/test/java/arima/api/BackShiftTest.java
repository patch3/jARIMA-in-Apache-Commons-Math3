package arima.api;

import arima.api.models.BackShift;
import org.junit.Test;

import static org.junit.Assert.assertTrue;


public class BackShiftTest {
    @Test
    public void testApplyOperator() {
        BackShift op1 = new BackShift(2, true);
        BackShift op2 = new BackShift(1, true);
        BackShift merged = op1.apply(op2);
        assertTrue(merged.getDegree() >= 3);
    }
}