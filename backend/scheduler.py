
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from datetime import datetime
from backend.resource_monitor import ResourceMonitor
from backend.feedback_logger import FeedbackLogger  # <— include this import

class FineTuneScheduler:
    def __init__(self, interval=600, log_dir="./data/logs"):
        self.monitor = ResourceMonitor()
        self.interval = interval
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "scheduler.log")

    def log(self, message, console=False):
        """Write logs to file silently."""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_line = f"{timestamp} {message}\n"
        if console:
            print(log_line.strip())
        with open(self.log_file, "a") as f:
            f.write(log_line)

    def run(self):
        self.log("🚀 Fine-tune scheduler started.", console=True)
        feedback_logger = FeedbackLogger()

        while True:
            try:
                idle, stats = self.monitor.is_system_idle()
                status = f"CPU={stats['cpu']}% | MEM={stats['mem']}% | TEMP={stats['temp']}°C"
                if stats["battery"] is not None:
                    status += f" | Battery={stats['battery']}%"
                else:
                    status += " | Battery=N/A"

                feedback_data = feedback_logger.get_feedback_dataset()
                if len(feedback_data) >= 5:
                    self.log(f"🧠 {len(feedback_data)} feedback records found. Triggering reinforcement fine-tune.")
                    os.system("python backend/reward_trainer.py >> data/logs/fine_tune.log 2>&1")
                else:
                    self.log(f"📊 Only {len(feedback_data)} feedback samples — skipping reinforcement fine-tune.")

                if idle:
                    self.log(f"✅ System idle ({status}) — starting fine-tuning.")
                    os.system("python backend/finetuner.py >> data/logs/fine_tune.log 2>&1")
                    self.log("🏁 Fine-tuning completed. Next check in 10 min.")
                else:
                    self.log(f"⚙️ System busy ({status}) — skipping fine-tuning.")
            except Exception as e:
                self.log(f"⚠️ Scheduler error: {e}")

            time.sleep(self.interval)
