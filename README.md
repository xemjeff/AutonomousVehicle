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
- GPS
- Ping ))) Distance
- <b>Speaker</b> <i>(bluetooth or separate amp too)</i>
- <b>IMU</b> <i>(code integration)</i>
- <b>Wheel Encoding</b> <i>(maybe not necessary)</i>
- Battery Voltage Level

<i>VISUAL SYSTEM</i>
+ Haarcascade Detection (36 objs.)
  - Self Calibration
  - Dynamic Loading
  - Road Sign Detection
+ Tensorflow Detection (20 objs.)
+ Stereographic Vision 
  - Depth Perception
- <b>Facial Recognition</b> <i>(if wanted)</i>
- Motion Detection
- OCR Alphanumeric Recognition
- Florescent Ball Recognition
- <b>Lane Detection</b> <i>(annotations need to be interpreted)</i>

<i>AUDIO SYSTEM</i>
- TTS from Secondary Server, scp to rPi's speaker
- Speech-to-text from rPi's microphone in gstreamer to Sphynx on Secondary Server

<i>NLP SYSTEM</i>
+ Modified AIML
  - <b>Specific Maker Faire Q&As</b> <i>(coded by team)</i>
  - Basic Conversation
- Commands Processing


<b><i>COMBINED SYSTEMS</i></b>
- Process EVERYTHING offline between rPi & Secondary Server via Wifi Relay
- Track & follow objects of interest via Visual, IMU & Wheel encoding
- Avoid hitting people's feet by Ping ))) sensor
- Engage with people around it via NLP & Audio
- Stay within a specific location via GPS
- Inform when battery is low & shutdown via Power
+ Perform Maker Faire specific tricks
  - React to Road Signs
  - Follow balls being led around 
  - Track and follow a lane
  - Play croquet with itself
- Try not to kill all humans


<b>SETUP:</b>

~ Run AutonomousVehicle/src/python/master.py on the RPi 3
+ master.py
  - /sensors/gps_monitor.py
  - /sensors/voltage_monitor.py
  - /visual/remote/server.py
  - /nlp/nlpServer.py
  + /controls/motion.py
    - /controls/ping_monitor.py
  - /audio/serveMicrophone.py

~ Run the following on the server 
- ../shell/startAPBridge.sh
- /visual/videorecClient.py
- /audio/ttsServer.py
- /audio/continuous.py

GPIO PINOUTS: (rPi3)
- GPS Monitor (GND),(VCC-5v),(TX-PIN8),(RX-PIN10)
- Self Volt Monitor (GND),(VDD-3.3v),(SDA-PIN3),(SCL-PIN5)

