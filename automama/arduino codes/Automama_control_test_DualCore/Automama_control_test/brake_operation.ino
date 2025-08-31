int brakeRead() {
  int raw = analogRead(lresPin);  // 0–4095 (on ESP32 ADC)
  float normalized = raw / 4095.0;
  float linearValue = pow(10, normalized) - 1;  // Range ~0–9
  int scaled = (int)(linearValue * 25);  // ~0 to 225
  return constrain(scaled, 0, 255);  // Final brake value
}

void brakeOp(byte target) {
  static bool brakeDone = false;
  static byte newTarget = 0;

  // Reset the 'done' flag if a new target is received.
  if(newTarget != target) {
    brakeDone = false;
  }
  
  // Safety feature: if brakes are applied, stop the throttle.
  if (brakeRead() < 20){
    stopThrottle();
  }

  // If the current brake position is less than the target, release the brakes.
  if(brakeRead() < target){
    if(!brakeDone){
      brakeRelease();
    }
  }
  // If the current brake position is more than the target, apply the brakes.
  else if(brakeRead() > target){
    if(!brakeDone){
      brakeApply();
    }
  }
  // If the target is reached, stop the brake motor.
  else{
    brakeStop();
    brakeDone = true;
    newTarget = target;
  }
}

void brakeApply(){
  analogWrite(brakeH, 125);
  analogWrite(brakeL, 0);
}

void brakeRelease(){
  analogWrite(brakeH, 0);
  analogWrite(brakeL, 125);
}

void brakeStop(){
   analogWrite(brakeH, 0);
   analogWrite(brakeL, 0);
}
