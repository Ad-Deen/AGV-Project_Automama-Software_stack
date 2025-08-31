void handleReception() {
  byte buffer[3];
  if (Serial1.available() >= 3) {
    Serial1.readBytes(buffer, 3);

    const int STEERING_OFFSET = 30;
    const int STEERING_LIMIT = 25;
    int decodedSteering = constrain((buffer[1] - STEERING_OFFSET), -STEERING_LIMIT, STEERING_LIMIT);

    if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
      targetThrottle = buffer[0];
      targetSteering = decodedSteering;
      targetBrake = buffer[2];
      xSemaphoreGive(dataMutex);
    }
  }
}

void handleTransmission() {
  byte response[3];
  int localSteeringPos;
  byte localBrakePos, localThrottleTarget;

  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localThrottleTarget = targetThrottle;
    localSteeringPos = steeringPosition;
    localBrakePos = currentBrakePosition;
    xSemaphoreGive(dataMutex);
  }

  const int STEERING_OFFSET = 30;
  response[0] = localThrottleTarget;
  response[1] = constrain(localSteeringPos, -30, 30) + STEERING_OFFSET;
  response[2] = localBrakePos;

  Serial1.write(response, 3);
}
