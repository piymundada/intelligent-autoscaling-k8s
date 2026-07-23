FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/spring-workload-simulator-0.0.1-SNAPSHOT.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", \
  "-Xms256m", "-Xmx512m", \
  "-XX:+UseG1GC", \
  "-Xlog:gc*:file=/tmp/gc.log:time,uptime:filecount=5,filesize=10m", \
  "-jar", "app.jar"]
