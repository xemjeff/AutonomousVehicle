# AutonomousVehicle
Code for the Burlington Robotics 1/10 Scale Autonomous Car Project

This code is for running a Traxxas Stampede modified RC car from the Raspberry Pi 3.
A mix of C, C++ and Python code.


<b>CONTAINS:</b>
<i>Bold items are those in which more work is needed</i>

Actuator Control:
- Car steering
- Car speed
- 2 Degrees of freedom pan/tilt

Sensory Input:
- Camera(s)
- Microphone
- Speaker
- IMU
- Wheel Encoding (not done)
- Battery Voltage Level

<i>VISUAL SYSTEM</i>
+ Haarcascade Detection (36 objs.)
  - Self Calibration
  - Dynamic Loading
  - Road Sign Detection
+ Tensorflow Detection (20 objs.)
+ Stereographic Vision 
  - Depth Perception
- <b>Facial Recognition (if wanted)</b>
- Motion Detection
- OCR Alphanumeric Recognition
- Florescent Ball Recognition

<i>AUDIO SYSTEM</i>
- TTS from Secondary Server, scp to rPi's speaker
- Speech-to-text from rPi's microphone in gstreamer to Sphynx on Secondary Server

<i>NLP SYSTEM</i>
+ Modified AIML
  - <b>Specific Maker Faire Q&As</b>
  - Basic Conversation
- Commands Processing


<b><i>COMBINED SYSTEMS</i></b>
- Process EVERYTHING offline between rPi & Secondary Server via Wifi Relay
- Track & follow objects of interest via Visual, IMU & Wheel encoding
- Avoid hitting people's feet by Visual disparity & motion detection
- Engage with people around it via NLP & Audio
- Stay within a specific location via GPS
- Inform when battery is low & shutdown via Power
+ Perform Maker Faire specific tricks
  - React to Road Signs
  - Follow balls being led around 
  - Play croquet with itself
- Try not to kill all humans


<b>STRUCTURE:</b>

(Secondary Server)
- ../shell/startAPBridge.sh
+ /controls/imageMotionHAL.py
  - /visual/videorecClient.py
- /audio/ttsServer.py
- /audio/continuous.py

(rPi 3)
+ /sensors/master.py
  - /sensors/gps_monitor.py
  - /sensors/voltage_monitor.py
  - /visual/remote/server.py
  - /nlp/nlpServer.py
  - /controls/motion.py
  - /audio/serveMicrophone.py


<b>REQUIREMENTS & SETUP:</b>

CONTROLS SETUP: (not done)
- Uses controls/motionController.py on rPi

WIFI RELAY SETUP: (done)
- Start ./src/shell/startAPBridge.sh on Secondary Server

GPS SETUP: (done)
- GPS sensor monitor requires 'gpsd /dev/ttyAMA0' to be run on rPi
- Uses sensors/gps_monitor.py on the rPi

POWER SETUP: (done)
- Uses sensors/voltage_monitor.py on the rPi

VISUAL SETUP: (almost done)
- Uses visual/remote/server.py on the rPi
- Start controls/imageMotionHAL.py on Secondary Server

AUDIO SETUP: (done)
- Start audio/ttsServer.py on Secondary Server (tts)
- Uses audio/serveMicrophone.py on rPi
- Start audio/continuous.py on Secondary Server (stt)

NLP SETUP: (needs maker faire Q&As)
- Uses nlp/nlpServer.py on rPi

GPIO PINOUTS: (rPi3)
- GPS Monitor (GND),(VCC-5v),(TX-PIN8),(RX-PIN10)
- Self Volt Monitor (GND),(VDD-3.3v),(SDA-PIN3),(SCL-PIN5)

