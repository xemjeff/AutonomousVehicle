#define NEUTRAL 1500
#define REVERSE 1000
#define FORWARD 2000
#define LEFT 0.75
#define RIGHT -0.75
#define SPEED_INCREMENT 0.1

//the defines are BCM pins, which is what the PiGPIO library references
#define GPIO_SPEED_CHANNEL 19 //board pin 35 (GPIO.24 Pi)
#define GPIO_STEER_CHANNEL 13 //board pin 33 (GPIO.23 Pi)

class CarControl
{
public:
	CarControl();
	virtual ~CarControl();

	void setDirection(float dir);
	float direction();

	void setSpeed(float s);
	float speed();

	void neutral();
	void faster();
	void slower();

	void left();
	void right();

	bool isReady();

private:
	float rangeLimit(float f);
	int toPWM(float num);

	void openGPIO();
	void closeGPIO();

private:
	float m_speed;
	float m_direction;
	bool m_ready;
};
