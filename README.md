# jARIMA in Apache Commons Math3

## Overview

This a library that develops an auto-ARIMA function in Java based on the article
by [Hyndman and Khandakar, 2008](https://www.jstatsoft.org/article/view/v027i03/v27i03.pdf) and implementation
in [R](https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/auto.arima) as in
the [original repository](https://github.com/O1sims/jARIMA).

## Technology stack

The library uses [ApacheCommonMath3](https://mvnrepository.com/artifact/org.apache.commons/commons-math3) as the backend
framework, [R](https://www.r-project.org/) as the comparable analytics engine with the
daemonised [Rserve](https://www.rforge.net/Rserve/) client to process time series data
and [Angular](https://angular.io/) as the frontend framework. Documentation of the RESTful API service is handled
by [Swagger](https://swagger.io/). Development is done within a [Docker](https://www.docker.com/). Note the we also
use [NGINX](https://www.nginx.com/) as the a load balancing reverse proxy.

## Accuracy

The `R` script written in `./analysis/R/compareARIMA.R` is used to test the accuracy and time taken for the ARIMA
analysis.

In terms of accuracy, the Java and R results converge as the time series data being supplied increases in size.

It is 10 to 100 times faster than the R implementation.

