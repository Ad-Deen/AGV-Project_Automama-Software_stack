// Debug control - must match main file setting
extern const bool DEBUG;

// Make global objects and handles visible to this file
extern SemaphoreHandle_t dataMutex;
extern ESP32Encoder encoder;
extern ESP32Encoder steeringEncoder;

void initResetButton() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
}
// Non-blocking manual steering function, now thread-safe.
void manualSteering() {
  static bool lastButtonState = HIGH;
  static unsigned long buttonPressTime = 0;
  static int manualTargetPosition = 0;

  byte reading = digitalRead(RESET_BUTTON);

  // Button press detection
  if (reading == LOW && lastButtonState == HIGH) {
    buttonPressTime = millis();
    if (DEBUG) Serial.println("Button pressed");
  }
  
  // Button release detection
  if (reading == HIGH && lastButtonState == LOW) {
    unsigned long pressDuration = millis() - buttonPressTime;
    
    // --- Long press: Toggle manual control ---
    if (pressDuration >= 1000) {
      bool isManual; // Local variable to hold the new state
      
      // Safely read and then toggle the manualSteeringActive flag
      if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
        manualSteeringActive = !manualSteeringActive;
        isManual = manualSteeringActive;
        
        if (isManual) {
          // When activating, set the manual encoder to the current steering position.
          // The readSteeringPosition() function is already thread-safe.
          readSteeringPosition(); 
          manualTargetPosition = steeringPosition;
          encoder.setCount(steeringPosition * 2);
        }
        xSemaphoreGive(dataMutex);
      }

      if (isManual) {
        Serial.println("Manual steering ACTIVATED");
      } else {
        steeringStop();
        Serial.println("Manual steering DEACTIVATED");
      }

    // --- Short press: Reset steering center (only if manual mode is active) ---
    } else {
      bool isManual;
      if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
        isManual = manualSteeringActive;
        xSemaphoreGive(dataMutex);
      }

      if (isManual) {
        steeringEncoder.setCount(0);
        encoder.setCount(0); // Also reset manual encoder
        manualTargetPosition = 0;
        steeringStop();
        saveSteeringPosition(); // This function must also be thread-safe
        Serial.println("Steering reset to zero!");
      }
    }
  }
  
  lastButtonState = reading;
  
  // --- Handle steering control if in manual mode ---
  bool isManualNow;
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    isManualNow = manualSteeringActive;
    xSemaphoreGive(dataMutex);
  }

  if (isManualNow) {
    manualTargetPosition = (int)(encoder.getCount() / 2);
    
    if (DEBUG) {
      int currentPos;
      if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
        currentPos = steeringPosition;
        xSemaphoreGive(dataMutex);
      }
      Serial.print(" | Manual Target: ");
      Serial.print(manualTargetPosition);
      Serial.print(" | Current: ");
      Serial.println(currentPos);
    }
    
    // Control steering motor directly in manual mode
    steeringControl(manualTargetPosition);
  }
}
