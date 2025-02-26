import depthai as dai

class PerformanceMonitor:
    def __init__(self, pipeline):
        self.system_logger = pipeline.create(dai.node.SystemLogger)
        self.system_logger.setRate(100)  # Update rate in Hz

        self.xout_system = pipeline.create(dai.node.XLinkOut)
        self.xout_system.setStreamName("system_logger")

        self.system_logger.out.link(self.xout_system.input)

    def get_performance_data(self, system_queue):
        system_data = system_queue.get()

        return {
            "CPU Usage": f"{system_data.leonCssCpuUsage.average * 100:.2f}%",
            "Memory Used": f"{system_data.ddrMemoryUsage.used / (1024 * 1024):.2f} MiB",
            "Chip Temperature": f"{system_data.chipTemperature.average:.2f}°C",
        }

