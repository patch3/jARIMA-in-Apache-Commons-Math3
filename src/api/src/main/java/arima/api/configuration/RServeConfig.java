package arima.api.configuration;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.validation.constraints.NotNull;

@Setter
@Configuration
@EnableConfigurationProperties
@ConfigurationProperties(prefix = "rserve")
public class RServeConfig {

    @NotNull
    private String hostname;


    @NotNull
    @Getter
    private int port;

    @Bean
    public String getHostname() {
        return this.hostname;
    }

}
