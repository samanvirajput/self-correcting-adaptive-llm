import psutil
import platform
import subprocess

class ResourceMonitor:
    def __init__(self, cpu_threshold=40, ram_threshold=60, battery_min=40):
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.battery_min = battery_min

    def get_system_load(self):
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        return cpu, mem

    def get_battery_status(self):
        battery = psutil.sensors_battery()
        if battery:
            return battery.percent, battery.power_plugged
        return None, True  # Assume safe if no battery (e.g. desktop)

    def get_temperature(self):
        try:
            if platform.system() == "Darwin":
                temp_output = subprocess.check_output(["istats", "cpu", "temp"]).decode()
                temp = float(temp_output.split(":")[1].split("°")[0].strip())
                return temp
            elif platform.system() == "Linux":
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    return temps["coretemp"][0].current
            return 40.0
        except Exception:
            return 40.0

    def is_system_idle(self):
        cpu, mem = self.get_system_load()
        battery, plugged = self.get_battery_status()
        temp = self.get_temperature()

        conditions = [
            cpu < self.cpu_threshold,
            mem < self.ram_threshold,
            (battery is None or battery > self.battery_min or plugged),
            temp < 80  # °C
        ]
        return all(conditions), {"cpu": cpu, "mem": mem, "battery": battery, "temp": temp}
