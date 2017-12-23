import os, pickle, time, signal
import pandas as pd
from datetime import datetime

class SignalHandler:
	def __init__(self):
		signal.signal(signal.SIGINT, self.stop_script)
		signal.signal(signal.SIGTERM, self.stop_script)

	def stop_script(self,signum,frame):
		print("Exiting Script")
		exit()

def generate_command(ip: str,block_size: str) -> str:
    command = 'nmap -n ' + ip + '/' + block_size
    return command

def perform_scan(command: str) -> pd.DataFrame:
    stdout = os.popen(command).read()
    stdout_lines = stdout.split("\n")
    start_time = ""
    active_hosts_count = 0
    scan_time = 0
    for x in stdout_lines:
        if x.startswith("Starting Nmap"):
            start_time = datetime.strptime((x[x.find("at ")+3:]), '%Y-%m-%d %H:%M %Z')
        if x.startswith("Nmap done"):
            active_hosts_count = int(x[x.find("(")+1:x.find(" host")])
            scan_time = float(x[x.find(" in ")+4:x.find(" second")-1])
    scan_results = pd.DataFrame(pd.Series({'Scan Start Time': start_time, 'Active Hosts': active_hosts_count, 'Scan Time': scan_time})).transpose().set_index('Scan Start Time')
    return scan_results

def start_scan(hostel=' - ') -> None:
    try:
        all_scan_results = pd.read_csv('scan_results.csv')
        all_scan_results = all_scan_results.set_index('Scan Start Time')
    except FileNotFoundError:
        all_scan_results = pd.DataFrame()
    scan_results = perform_scan(generate_command(ip,block_size))
    scan_results['IP Address'] = ip
    scan_results['Block Size'] = block_size
    scan_results['Hostel'] = hostel
    scan_results = scan_results.reindex(columns=['IP Address','Block Size','Active Hosts','Scan Time','Hostel'])
    print(scan_results)
    all_scan_results = pd.concat([all_scan_results, scan_results], axis=0)
    all_scan_results.to_csv('scan_results.csv')

all_hostel_ips = [
    ("Shivalik", "10.201.136.0", "23"),
    ("Vindyachal", "10.202.140.0", "23"),
    ("Jwalamukhi", "10.203.148.0", "23"),
    ("Aravali", "10.204.152.0", "23"),
    ("Karakoram", "10.205.156.0", "23"),
    ("Nilgiri", "10.206.160.0", "23"),
    ("Kailash", "10.207.164.0", "23"),
    ("Himadri", "10.242.172.0", "23"),
    ("Kumaon", "10.243.144.0", "22"),
    ("Girnar", "10.249.208.0", "22"),
    ("Udaigiri", "10.250.212.0", "22"),
    ("Zanskar", "10.251.216.0", "23"),
    ("Satpura", "10.252.220.0", "23")]

if __name__ == '__main__':
    handler = SignalHandler()
    scan_count = 0
    _ = [print(x[0]," - ",x[1]," - ",x[2]) for x in all_hostel_ips]
    ip = input("Enter the IP Address (Enter -1 to scan all IITD Hostel IPs): ")
    if ip != "-1":
        block_size = input("Enter the Block Size: ")
    while(True):
        scan_count = scan_count+1
        print("Scan Count - ",scan_count)
        if ip != "-1":
            hostel = " - "
            for x in all_hostel_ips:
                if x[1] == ip:
                    hostel = x[0]
                    break
            start_scan(hostel)
        else:
            for hostel, ip, block_size in all_hostel_ips:
                start_scan(hostel)
        time.sleep(1800)