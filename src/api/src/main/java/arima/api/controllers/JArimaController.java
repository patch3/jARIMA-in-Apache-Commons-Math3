package arima.api.controllers;

import arima.api.analytics.Arima;
import arima.api.models.ForecastResultModel;
import arima.api.models.TimeSeriesModel;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;


@RestController
@CrossOrigin
@RequestMapping("/j-arima")
public class JArimaController {

    @RequestMapping(
            value = "/",
            method = RequestMethod.POST)
    public ForecastResultModel calculateRArima(
            @Valid @RequestBody TimeSeriesModel rArima)
            throws Exception {

        ForecastResultModel forecastResult = Arima.forecast_arima(
                rArima.getTsData(), rArima.getForecastPeriod());

        return forecastResult;
    }
}
