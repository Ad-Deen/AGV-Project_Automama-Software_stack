extern const bool DEBUG;

// Make the global mutex handle visible to this file
extern SemaphoreHandle_t dataMutex;
extern ESP32Encoder steeringEncoder; // Make encoder visible

void readSteeringPosition() {
  long newSteeringPos = -steeringEncoder.getCount();

  // Acquire the mutex before writing to the shared global variable.
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    steeringPosition = newSteeringPos;
    xSemaphoreGive(dataMutex);
  }
}

void steeringControl(int target) {
  int localSteeringPos;
  
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localSteeringPos = steeringPosition;
    xSemaphoreGive(dataMutex);
  }

  if(target > 25) target = 25;
  else if(target < -25) target = -25;

  // Safety check to stop the motor if it's too far off or outside limits.
  if(abs(localSteeringPos) > 25 || abs(localSteeringPos - target) > 5) {
    steeringStop();
  }
  else{
    // Move the motor toward the target.
    if (localSteeringPos > target) {
      steeringLeft();
      if (DEBUG) Serial.println("LEFT");
    } else if (localSteeringPos < target) {
      steeringRight();
      if (DEBUG) Serial.println("RIGHT");
    } else {
      steeringStop();
      if (DEBUG) Serial.println("STOPPED");
    }
  }
}

// Moves the steering to the calibrated center position.
void gotoCenter() {
  Serial.println("Moving steering to center position (0)...");
  int lastPos;
  int currentPos;

  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
      currentPos = steeringPosition;
      xSemaphoreGive(dataMutex);
  }
  
  while (currentPos != 0) {
    lastPos = currentPos;
    Serial.print("Last position was: ");
    Serial.println(lastPos);

    if (currentPos > 0) {
      steeringLeft();
      Serial.print("Moving LEFT - Current: ");
    } else if (currentPos < 0) {
      steeringRight();
      Serial.print("Moving RIGHT - Current: ");
    }
    
    // Get the latest position for the next loop iteration
    if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
        currentPos = steeringPosition;
        xSemaphoreGive(dataMutex);
    }
    Serial.println(currentPos);
    delay(100);
  }
  
  steeringStop();
  Serial.println("Steering centered at position 0");
  
  saveSteeringPosition(); // This function should also be thread-safe
}

void steeringLeft() {
  analogWrite(Steering_RPWM, 0);
  analogWrite(Steering_LPWM, 255);
}

void steeringRight() {
  analogWrite(Steering_RPWM, 255);
  analogWrite(Steering_LPWM, 0);
}

void steeringStop() {
  analogWrite(Steering_RPWM, 0);
  analogWrite(Steering_LPWM, 0);
}

void initSteeringPins() {
  pinMode(Steering_RPWM, OUTPUT);
  pinMode(Steering_LPWM, OUTPUT);
  steeringStop();
}
