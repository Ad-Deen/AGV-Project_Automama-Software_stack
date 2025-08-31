
void throttleControl(byte speed) {
  // speed = map(speed, 0, 10, 0, 255);
  if (speed < 0) speed = 0;
  if (speed > 200) speed = 200;
//  isBrake = 0;
//  breakRead() < 9? isBrake = 0: isBrake = 1;
//  if (!isBrake) {
  ledcWrite(throttle, speed); 
  // analogWrite(throttle, speed);
//  }
}

void stopThrottle() {
  throttleControl(0);
}
