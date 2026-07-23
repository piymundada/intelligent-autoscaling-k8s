package com.example.springworkloadsimulator.config;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.distribution.DistributionStatisticConfig;
import org.springframework.boot.actuate.autoconfigure.metrics.MeterRegistryCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * Enables histogram buckets and client-side percentiles for HTTP server requests.
 * Required for p95/p99 quantile queries in Prometheus (histogram_quantile).
 * Configured programmatically because Spring Boot 3.2 changed the Observation API
 * and property-based histogram config no longer applies automatically.
 */
@Configuration
public class MetricsConfig {

    @Bean
    public MeterRegistryCustomizer<MeterRegistry> httpHistogramCustomizer() {
        return registry -> registry.config().meterFilter(
            new MeterFilter() {
                @Override
                public DistributionStatisticConfig configure(
                        io.micrometer.core.instrument.Meter.Id id,
                        DistributionStatisticConfig config) {

                    // Enable histograms for HTTP server requests and load endpoints
                    if (id.getName().startsWith("http.server.requests")) {
                        return DistributionStatisticConfig.builder()
                            .percentilesHistogram(true)           // enables _bucket metrics
                            .percentiles(0.5, 0.95, 0.99)        // client-side gauges
                            .minimumExpectedValue(Duration.ofMillis(1).toNanos())
                            .maximumExpectedValue(Duration.ofSeconds(10).toNanos())
                            .build()
                            .merge(config);
                    }
                    return config;
                }
            }
        );
    }
}
