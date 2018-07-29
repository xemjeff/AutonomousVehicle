# Finds the average mV
# Returns time left until minimum sustainable voltage

from time import sleep
import redis
import numpy as np
import pandas as pd
import Adafruit_ADS1x15

memory = redis.StrictRedis(host='localhost',port=6379,db=0)

def trendline(data, order=1):
    coeffs = np.polyfit(data.index.values, list(data), order)
    slope = coeffs[-2]
    return float(slope)

arrayVolt = [0,0,0,0,0,0,0,0,0,0]
trendVolt = []
i=0
while True:
        trendVolt.append(i)
        if len(trendVolt)==60:
            break
trendList = list(range(60))

adc = Adafruit_ADS1x15.ADS1115()
GAIN = 1
adc.start_adc(0, gain=GAIN)

try:
    while True:
        value = adc.get_last_result()
        arrayVolt[0] = value
        trendVolt[0] = value
        for count in xrange(1,len(arrayVolt)): arrayVolt[10-count]=arrayVolt[9-count]
        for count in xrange(1,len(trendVolt)): trendVolt[60-count]=trendVolt[59-count]
        if 0 in arrayVolt: final=arrayVolt[1]
        else: final = float(sum(arrayVolt))/len(arrayVolt)
        memory.set('voltage', final)
        if (not 0 in trendVolt):
            df = pd.DataFrame({'time': trendList, 'voltage': trendVolt[::-1]})
            slope = trendline(df['voltage'])
            if slope<-0.7:
                time_left = (final-min)*-slope
                memory.set('time_left',time_left)
        sleep(1)
except:
    adc.stop_adc()
