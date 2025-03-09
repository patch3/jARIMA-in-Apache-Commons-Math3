package arima.api.services;

import org.rosuda.REngine.Rserve.RConnection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.validation.constraints.NotNull;


@Service
public class RArimaService {

    private final Logger LOGGER = LoggerFactory.getLogger(RArimaService.class);
    @Value("${rserve.hostname}")
    @NotNull
    private String hostname;
    @Value("${rserve.port}")
    @NotNull
    private int port;
    private RConnection connection;

    public void performArima(int[] tsData) throws Exception {

        LOGGER.info("Connecting to Rserve: {}:{}", this.hostname, this.port);
        this.connection = new RConnection(
                this.hostname,
                this.port);
        this.connection.assign("tsData", tsData);
        this.connection.voidEval("dffv <- data.frame(forecast::forecast(forecast::auto.arima(tsData), 1))");
        final double hi95 = this.connection.eval("dffv$Hi.95").asDouble();
        final double lo95 = this.connection.eval("dffv$Lo.95").asDouble();
        LOGGER.info(Double.toString(hi95));
        LOGGER.info(Double.toString(lo95));
    }

}