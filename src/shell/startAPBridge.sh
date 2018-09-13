#!/bin/sh

echo "*Stopping Services*"
nmcli radio wifi off
rfkill unblock wlan
killall wpa_supplicant
killall dhclient
service hostapd stop
service udhcpd stop

echo "Starting up host wifi"
ifconfig wlan0 up

echo "*Starting AP*"
iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
hostapd -d /etc/hostapd/hostapd.conf &
service udhcpd start
sysctl net.ipv4.ip_forward=1
