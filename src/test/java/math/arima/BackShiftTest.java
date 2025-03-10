package math.arima;

import math.arima.models.BackShift;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;


public class BackShiftTest {
    @Test
    public void testApplyOperator() {
        BackShift op1 = new BackShift(2, true);
        BackShift op2 = new BackShift(1, true);
        BackShift merged = op1.apply(op2);
        assertTrue(merged.getDegree() >= 3);
    }
}