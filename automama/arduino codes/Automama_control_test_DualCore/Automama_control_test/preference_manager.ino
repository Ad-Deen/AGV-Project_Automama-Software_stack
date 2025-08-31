void saveSteeringPosition() {
  long localSteeringPos;

  // Safely read the global steeringPosition into a local variable.
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localSteeringPos = steeringPosition;
    xSemaphoreGive(dataMutex);
  }
  
  // Perform the non-volatile memory operation.
  preferences.begin("steering", false);
  preferences.putLong("position", localSteeringPos);
  preferences.end();

  if (DEBUG) {
    Serial.print("Saved steering position: ");
    Serial.println(localSteeringPos);
  }
}

// Function to load steering position from preferences.
// This is typically called from setup() before tasks start, so a mutex isn't
// strictly required here, but it's good practice for functions accessing shared resources.

void loadSteeringPosition() {
  preferences.begin("steering", true);
  long savedPosition = preferences.getLong("position", 0);
  // Safety check to prevent loading an extreme value
  if(abs(savedPosition) > 30) savedPosition = 0; 
  preferences.end();
  
  // Set the hardware encoder to the saved position.
  steeringEncoder.setCount(savedPosition * (-1));
  
  // Update the global variable. readSteeringPosition() is already thread-safe.
  readSteeringPosition(); 
  
  if (DEBUG) {
    long currentPos;
    // Safely read the position for printing
    if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
      currentPos = steeringPosition;
      xSemaphoreGive(dataMutex);
    }
    Serial.print("Loaded steering position: ");
    Serial.println(currentPos);
  }
}

// Function to periodically save steering position.
void periodicSave() {
  static unsigned long lastSaveTime = 0;
  static long lastSavedPosition = 0;
  long localCurrentPos;

  // Safely get the current position to check if it has changed.
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localCurrentPos = steeringPosition;
    xSemaphoreGive(dataMutex);
  }
  
  // Save every 1 second if position has changed.
  if (millis() - lastSaveTime > 1000 && localCurrentPos != lastSavedPosition) {
    saveSteeringPosition(); // This function is now thread-safe.
    lastSaveTime = millis();
    lastSavedPosition = localCurrentPos;
  }
}
