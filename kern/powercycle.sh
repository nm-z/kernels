#!/bin/bash
# powercycle.sh: cold-cycle archy (power it off, wake it with a magic packet). Run it on engi, not on archy.
# A cold boot resets all eight GPUs and applies pending ECC mode changes; use it when `nvidia-smi -r` hangs in the driver.
# Tested 2026-10-16: the 45 s guard powered archy off (sync, SysRq u/o) while its driver was hung, and a magic packet woke it from S5.
# WoL on archy: Wake-on: g + BIOS token enabled (skill wake-device); the broadcast goes out engi's wired NIC.
# Stop first: holders, flashcore, flashd (serve.sh stop), and tell whoever else uses the box (codex workers, the llama-server
# services on dies 6/7): everything on archy ends.
# powercycle.sh [archy|sentry]: the host's MAC and address; the broadcast goes out engi's wired NIC either way
HOST=${1:-archy}; BCAST=192.168.1.255
case $HOST in
	archy)  MAC=98:90:96:bb:be:75; IP=192.168.1.218 ;;
	sentry) MAC=98:90:96:aa:0b:fc; IP=192.168.1.20 ;;      # Precision T7810 (BIOS A34, the F1 bypass driver on its ESP), enp0s25, Wake-on: g
	*) echo "powercycle: unknown host $HOST (archy or sentry)"; exit 1 ;;
esac

tick() { timeout 1 tail -f /dev/null; }   # one second (ping returns at once while the host answers, so a bare ping loop does not wait)

wol() {
	printf "%b" "$({ printf 'ffffffffffff'; for i in $(seq 16); do printf '%s' "${MAC//:/}"; done; } | sed 's/../\\x&/g')" \
		| socat -u - UDP4-DATAGRAM:$BCAST:9,broadcast
}

[ "$(ssh $HOST hostname)" = archy ] || { echo "powercycle: $HOST is not archy, refusing"; exit 1; }
echo "powercycle: powering $HOST off"
# Nothing may make the shutdown wait. Processes stuck in the NVIDIA driver (D state) cannot be killed, so systemd would sit on their
# stop jobs for 90 s each, and the driver's own shutdown hook can block on the lock that is hung. Two guards run as transient units with
# DefaultDependencies=no (the shutdown does not stop them): at 45 s, when the clean path has had its time, sync + remount read-only +
# SysRq power off; at 75 s SysRq reboot, which skips the device shutdown hooks (the box then comes back on its own). They are separate
# units because a write to /proc/sysrq-trigger that blocks never returns to the next command of its shell.
GUARD1='timeout 45 tail -f /dev/null; timeout 15 sync; echo u > /proc/sysrq-trigger; echo o > /proc/sysrq-trigger'
GUARD2='timeout 75 tail -f /dev/null; echo b > /proc/sysrq-trigger'
ssh $HOST "sudo -n systemd-run --quiet --unit=pc-guard1 --property=DefaultDependencies=no /bin/sh -c '$GUARD1'
sudo -n systemd-run --quiet --unit=pc-guard2 --property=DefaultDependencies=no /bin/sh -c '$GUARD2'
timeout 20 sync; sudo -n systemctl poweroff --no-block"
for i in $(seq 200); do ping -c1 -W1 $IP > /dev/null || break; tick; done
ping -c1 -W1 $IP > /dev/null && { echo "powercycle: $HOST is still up after 200 s (the guards did not fire)"; exit 1; }
echo "powercycle: $HOST is down, waking it"
# the packet is resent every 10 s until the host answers (the first one can arrive while the NIC is still powering down; if the
# guard rebooted the box it is already on its way up and the packet does no harm)
up=0
for i in $(seq 18); do
	wol
	if ping -c1 -W10 $IP > /dev/null; then up=1; break; fi
	tick
done
[ $up = 1 ] || { echo "powercycle: no answer from $HOST after 3 minutes (WoL not working? press the power button)"; exit 1; }
echo "powercycle: $HOST answers ping, waiting for ssh"
for i in $(seq 120); do ssh -o ConnectTimeout=2 -o BatchMode=yes $HOST true && break; tick; done
ssh $HOST uptime
