#include <stdio.h>
#include <pigpio.h>
#include "control.h"

CarControl::CarControl()
{
	m_speed = 0.0;
	m_direction = 0.0;
	m_ready = false;
	openGPIO();
}

CarControl::~CarControl()
{
	closeGPIO();
}


void CarControl::setDirection(float dir)
{
	m_direction = rangeLimit(dir);
	int pulseWidth = toPWM(m_direction);
    printf("-->> direction: %f  pulse: %d", m_direction, pulseWidth);
    gpioServo(GPIO_STEER_CHANNEL, pulseWidth);     
}

float CarControl::direction()
{
	return m_direction;
}

void CarControl::setSpeed(float sp)
{
	m_speed = rangeLimit(sp);
	int pulseWidth = toPWM(m_speed);
	printf("speed: %f  pulse: %d", m_speed, pulseWidth);
    gpioServo(GPIO_SPEED_CHANNEL, pulseWidth);     
}

float CarControl::speed()
{
	return m_speed;
}

void CarControl::neutral()
{
	setSpeed(0.0);
}

void CarControl::faster()
{
	float sp = speed() + SPEED_INCREMENT;
	setSpeed(sp);
}

void CarControl::slower()
{
	float sp = speed() - SPEED_INCREMENT;
	setSpeed(sp);
}

void CarControl::left()
{
	setDirection(LEFT);
}

void CarControl::right()
{
	setDirection(RIGHT);
}

float CarControl::rangeLimit(float f)
{
  if (f > 1.0) f = 1.0;
  if (f < -1.0) f = -1.0;
  return f;
}

int CarControl::toPWM(float num)
{
	return NEUTRAL + num*(FORWARD-NEUTRAL);
}

bool CarControl::isReady()
{
	return m_ready;
}

void CarControl::openGPIO()
{
	m_ready = (gpioInitialise() >= 0);
}

void CarControl::closeGPIO()
{
	gpioTerminate();   
}
