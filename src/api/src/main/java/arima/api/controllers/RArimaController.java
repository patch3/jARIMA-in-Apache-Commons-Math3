package arima.api.controllers;

import arima.api.models.RArimaResultModel;
import arima.api.models.TimeSeriesModel;
import org.rosuda.REngine.Rserve.RConnection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;


@RestController
@CrossOrigin
@RequestMapping("/r-arima")
public class RArimaController {

    private final Logger LOGGER = LoggerFactory.getLogger(RArimaController.class);
    private RConnection connection;
    @Value("${rserve.port}")
    @NotNull
    private int port;
    @Value("${rserve.hostname}")
    @NotNull
    private String hostname;

    @RequestMapping(
            value = "/",
            method = RequestMethod.POST)
    public RArimaResultModel calculateRArima(
            @Valid @RequestBody TimeSeriesModel rArima)
            throws Exception {
        LOGGER.info("Connecting to Rserve: {}:{}", this.hostname, this.port);
        this.connection = new RConnection(this.hostname, this.port);
        this.connection.assign("tsData", rArima.getTsData());
        this.connection.assign("forecastPeriod", String.valueOf(rArima.getForecastPeriod()));
        LOGGER.info("Evaluating time-series data...");
        this.connection.voidEval(
                "dffv <- data.frame(forecast::forecast(forecast::auto.arima(tsData), forecastPeriod))");

        return new RArimaResultModel(
                this.connection.eval("dffv$Point.Forecast").asDoubles(),
                this.connection.eval("dffv$Hi.95").asDoubles(),
                this.connection.eval("dffv$Lo.95").asDoubles());
    }
}
