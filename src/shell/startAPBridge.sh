#!/bin/sh

echo "*Stopping Services*"
nmcli radio wifi off
rfkill unblock wlan
airmon-ng check kill
killall wpa_supplicant
killall dhclient
service hostapd stop
service udhcpd stop
ifconfig wlan1 down

echo "Starting up host wifi"
ifconfig wlan0 up
wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
dhclient wlan0 -v

echo "*Setting up Routes"
sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
sudo iptables -A FORWARD -i wlan0 -o wlan1 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i wlan1 -o wlan0 -j ACCEPT
sleep 5
ifconfig wlan1 192.168.6.1
sleep 3

echo "*Starting AP*"
hostapd -d /etc/hostapd/hostapd.conf &
service udhcpd start
sysctl net.ipv4.ip_forward=1
