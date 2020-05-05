#ifndef __CONTROL_H_
#define __CONTROL_H_

#include "sys.h"
#include "motor.h"

#define CONTROL_LOOP_PERIOD_MS	        5
#define CONTROL_SAFETY_ANGLE_MIN        2148
#define CONTROL_SAFETY_ANGLE_MAX        3996
#define CONTROL_MOTOR_MAX_SPEED         ((int)(0.95 * MOTOR_FULL_SCALE + 0.5))

// Defaults
#define CONTROL_ANGLE_SET_POINT			3129 // for unit 2 from aliexpress // for unit 1 from ali 3383 // set it to exactly the vertical position of the pendulum, by printing values and then averaging them
#define CONTROL_ANGLE_SMOOTHING         1.0f        // 0 to 1.0 (1.0 disables smoothing)
#define CONTROL_ANGLE_KP                400.0f
#define CONTROL_ANGLE_KD                400.0f
#define CONTROL_ANGLE_AVERAGE_LEN		1  //16        // Number of samples to average over to determine angular displacement (max is 32)
#define CONTROL_POSITION_SET_POINT      0
#define CONTROL_POSITION_SMOOTHING      1 // 0.2f        // 0 to 1.0 (1.0 disables smoothing)
#define CONTROL_POSITION_KP             20.0f
#define CONTROL_POSITION_KD             300.0f
#define CONTROL_POSITION_PERIOD_MS      5 // 25

void CONTROL_Init(void);
void CONTROL_Calibrate(void);
void CONTROL_ToggleState(void);
void CONTROL_Loop(void);
void CONTROL_BackgroundTask(void);

#endif /*__CONTROL_H_*/