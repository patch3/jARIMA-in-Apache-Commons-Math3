package arima.api.controllers;

import arima.api.models.ForecastResultModel;
import arima.api.models.TimeSeriesModel;
import arima.api.analytics.timeseries.arima.Arima;

import javax.validation.Valid;

import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("/j-arima")
public class JArimaController {
	
	@RequestMapping(
			value = "/", 
			method = RequestMethod.POST)
	public ForecastResultModel calculateRArima(
			@Valid @RequestBody TimeSeriesModel rArima)
					throws Exception {
		
		
		
		ForecastResultModel forecastResult = Arima.forecast_arima(
				rArima.getTSData(), rArima.getForecastPeriod());
		
		return forecastResult; 
	}
}
