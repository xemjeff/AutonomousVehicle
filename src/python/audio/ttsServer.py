import os,redis,os.path,sys,paramiko
from time import sleep

try:
	ip = os.popen('nslookup raspberrypi').read().split('Address: ')[1].replace('\n','')
	memory = redis.StrictRedis(host=ip, port=6379, db=0)
except:
	print "The rPi is offline"
	sys.exit () 

while True:
	if memory.get('ttsOverride'):
		text = memory.get('ttsOverride')
		memory.set('ttsOverride','')
		if volume != 35:
			os.popen('amixer set Master 35% >/dev/null')
			volume = 35
		os.system('rm ~/.wine/drive_c/tts.wav')
		os.system('echo "%s" > ~/.wine/drive_c/text.txt' % text.replace('"',''))
		os.system('cd ~/.wine/drive_c && wine cscript "jampal/ptts.vbs" -voice "VW Bridget" -u "text.txt" -w "tts.wav"')
		while os.path.isfile('/root/.wine/drive_c/tts.wav') == False: sleep(0.1)
		# SSH file into rPi and mplayer it
		os.system('sshpass -p "password" scp /root/.wine/drive_c/tts.wav AV@%s:/tmp')
		ssh = paramiko.SSHClient()
		ssh.connect(server, username=username, password=password)
		ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("mplayer /tmp/tts.wav")
	sleep(0.1)

