#include <Arduino.h>
#include <ESP32Encoder.h>
#include <Preferences.h>

const bool DEBUG = false;

// --- Motor and Sensor Pins ---
#define Steering_RPWM 13 
#define Steering_LPWM 12
#define brakeH 10
#define brakeL 11
#define throttle 5
#define lresPin 4

// --- Communication Pins ---
#define RXD1 18
#define TXD1 17

// --- Manual Control Pins ---
#define CLK 20
#define DT 21
const int RESET_BUTTON = 45;

// --- Steering Encoder Pins ---
#define STEERING_CLK 16
#define STEERING_DT 15

volatile byte targetThrottle = 0;
volatile int targetSteering = 0;
volatile byte targetBrake = 36;
volatile int steeringPosition = 0;
volatile byte currentBrakePosition = 0;
volatile bool manualSteeringActive = false;

TaskHandle_t ControlTaskHandle, SensorTaskHandle, PeriodicSaveTaskHandle; // Added handle for save task
TaskHandle_t ReceiveTaskHandle, SendTaskHandle, ManualSteeringTaskHandle;
SemaphoreHandle_t dataMutex;
Preferences preferences; // Define preferences object globally
ESP32Encoder encoder;
ESP32Encoder steeringEncoder;

//================================================================
//  CORE 1 TASKS 🦾 (Communication & Manual Input)
//================================================================

void ReceiveLoop(void *pvParameters) {
  for (;;) {
    handleReception();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void SendLoop(void *pvParameters) {
  for (;;) {
    handleTransmission();
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

void ManualSteeringLoop(void *pvParameters) {
    for (;;) {
        manualSteering();
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

//================================================================
//  CORE 0 TASKS ⚙️ (Control, Sensing, & Saving)
//================================================================

void ControlLoop(void *pvParameters) {
  for (;;) {
    handleControl();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void SensorReadLoop(void *pvParameters) {
  for (;;) {
    handleSensorReading();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// New task to periodically save the steering position
void PeriodicSaveLoop(void *pvParameters) {
    for (;;) {
        periodicSave();
        // Run this check less frequently as it writes to flash memory
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200, SERIAL_8N1, RXD1, TXD1);
  encoder.attachHalfQuad(DT, CLK);
  steeringEncoder.attachHalfQuad(STEERING_DT, STEERING_CLK);
  encoder.setCount(0);
  // Initialize hardware, etc.
  loadSteeringPosition();
  initSteeringPins();
  initResetButton();

  dataMutex = xSemaphoreCreateMutex();

  // --- Create Core 1 Tasks ---
  xTaskCreatePinnedToCore(ReceiveLoop, "Receive", 10000, NULL, 3, &ReceiveTaskHandle, 1);
  xTaskCreatePinnedToCore(SendLoop, "Send", 10000, NULL, 2, &SendTaskHandle, 1);
  xTaskCreatePinnedToCore(ManualSteeringLoop, "Manual", 10000, NULL, 1, &ManualSteeringTaskHandle, 1);

  // --- Create Core 0 Tasks ---
  xTaskCreatePinnedToCore(ControlLoop, "Control", 10000, NULL, 2, &ControlTaskHandle, 0);
  xTaskCreatePinnedToCore(SensorReadLoop, "Sensors", 10000, NULL, 3, &SensorTaskHandle, 0);
  // Create the new low-priority save task on Core 0
  xTaskCreatePinnedToCore(PeriodicSaveLoop, "SavePos", 4096, NULL, 1, &PeriodicSaveTaskHandle, 0);


  vTaskDelete(NULL); // Delete the default Arduino loop
}

void loop() {}


void handleControl() {
  int localTargetSteering;
  byte localTargetBrake, localTargetThrottle;
  bool isManual;

  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    localTargetSteering = targetSteering;
    localTargetBrake = targetBrake;
    localTargetThrottle = targetThrottle;
    isManual = manualSteeringActive;
    xSemaphoreGive(dataMutex);
  }

  brakeOp(localTargetBrake);

  if (!isManual) {
    if (abs(localTargetSteering) <= 25) steeringControl((int)localTargetSteering);
  }

  if (localTargetThrottle <= 150 && localTargetThrottle >= 0) {
    throttleControl(localTargetThrottle);
  }
}

void handleSensorReading() {
  readSteeringPosition();
  byte newBrakePos = brakeRead();

  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    currentBrakePosition = newBrakePos;
    xSemaphoreGive(dataMutex);
  }
}
