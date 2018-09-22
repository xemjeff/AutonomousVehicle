import socket,paramiko,sys,os
sys.path.insert(0,'/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual')
print "Starting the Ball Tracking script for the AV\n"
print "Finding rPis IP with nslookup"
ip = ''
try:
	ip = os.popen('timeout 2 nslookup raspberrypi').read().split('Address: ')[1].replace('\n','')
except:
	print "Router doesn't support nslookup. Trying nmap (may take 10 seconds)"
	os = os.popen('uname -a').read()
	if 'Linux' in os: gw = os.popen('ip route | grep default').read().split('via ')[1].split(' dev')[0]
	else: gw = os.popen('netstat -nr | grep "^default"').read()
	try: 
		os.popen('nmap -sP '+'.'.join(gw.split('.')[:-1])+'.0/24').read().split('raspberrypi.home (')[1].split(')')[0]
	except: 
		print "Error with nmap"
		ip = raw_input("Input IP manually: ")
		pass
	if not '.' in ip: ip = raw_input("Input IP manually: ")
	pass
print "Aquired IP "+str(ip)
print "Checking that AV is online"
test = os.system('ping -c 1 '+ip)
if test != 0: 
	print "AV is offline - Maybe power is too low?"
	sys.exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex((ip,5000))
if result != 0:
	print "rPis camera server is starting"
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(ip, username='root', password='stop22')
	ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python /root/AutonomousVehicle/src/python/visual/remote/server.py &")
	ssh.close()
else: print "rPis camera server already started"

print "Starting the Video processing client"
from videorecClient import videorecClient
v = videorecClient()
print "Video client running"
print "Starting master.py on the rPi"
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip, username='root', password='stop22')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("pigpiod &")
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("python /root/AutonomousVehicle/src/python/master.py &")
ssh.close()
print "\nAV is ready for the ball! (make sure to turn on ESC too!)"


