// Make the global mutex handle visible to this file
extern SemaphoreHandle_t dataMutex;

// Safely prints the current sensor data to the Serial Monitor.
void read_sensor_data(){
  int localSteeringPos;
  
  // Safely read the shared steeringPosition variable
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localSteeringPos = steeringPosition;
    xSemaphoreGive(dataMutex);
  }

  Serial.print("Sensor -> SteeringPos: ");
  Serial.print(localSteeringPos);
  Serial.print(" | BrakePos: ");
  Serial.print(String(brakeRead())); // brakeRead() is self-contained
  Serial.println(" ");
}

// Safely prints the last received target data to the Serial Monitor.
void read_received_data(){
  byte localTargetThrottle;
  int localTargetSteering;
  byte localTargetBrake;

  // Safely read all shared target variables at once.
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localTargetThrottle = targetThrottle;
    localTargetSteering = targetSteering;
    localTargetBrake = targetBrake;
    xSemaphoreGive(dataMutex);
  }

  Serial.print("Target -> Throttle: ");
  Serial.print(localTargetThrottle);
  Serial.print(" | Steering: ");
  Serial.print(localTargetSteering);
  Serial.print(" | Brake: ");
  Serial.print(localTargetBrake);
  Serial.println(" ");
}
