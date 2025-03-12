package math.arima;

import lombok.val;
import math.series.time.arima.models.BackShift;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;


public class BackShiftTest {
    @Test
    public void testApplyOperator() {
        val op1 = new BackShift(2, true);
        val op2 = new BackShift(1, true);
        val merged = op1.apply(op2);
        assertTrue(merged.getDegree() >= 3);
    }
}