import subprocess
import time
import os
import glob
import logging
import sys
import shutil
import platform
import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | SUPERVISOR | %(message)s')
logger = logging.getLogger("Supervisor")

# Configuration
CYCLE_TIME_SECONDS = 15 * 60  # 15 Minutes
LEAD_TIME_SECONDS =  5* 60       # 5 Minutes (Optimize while bot is finishing)
BOT_SCRIPT = "main.py"
OPTIMIZER_SCRIPT = "scripts/optimize.py"
TRAINER_SCRIPT = "scripts/train_ml.py"
DATA_DIR = "data"
MAX_RETENTION = 10                # Keep last 10 files

IS_WINDOWS = platform.system() == "Windows"

def get_latest_capture_file():
    """Find the most recent capture file."""
    files = glob.glob(os.path.join(DATA_DIR, "capture_*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def cleanup_old_files():
    """Keep only the last N capture files."""
    files = glob.glob(os.path.join(DATA_DIR, "capture_*.jsonl"))
    if len(files) <= MAX_RETENTION:
        return
    
    # Sort by time (oldest first)
    files.sort(key=os.path.getmtime)
    
    # Delete oldest
    to_delete = files[:-MAX_RETENTION]
    for f in to_delete:
        try:
            os.remove(f)
            logger.info(f" Cleanup: Deleted old file {os.path.basename(f)}")
        except Exception as e:
            logger.error(f" Failed to delete {f}: {e}")

def run_process(command, wait=False, low_priority=False):
    """Run a subprocess."""
    # On Linux, use 'nice' to lower priority for training tasks
    if low_priority and not IS_WINDOWS:
        command = f"nice -n 19 {command}"
        
    try:
        if wait:
            logger.info(f" Running: {command}")
            subprocess.run(command, shell=True, check=True)
        else:
            logger.info(f" Launching: {command}")
            # Use Popen to run in background
            return subprocess.Popen(command, shell=True)
    except subprocess.CalledProcessError as e:
        logger.error(f" Command failed: {e}")
        return None

def kill_process(process):
    """Kill a process tree safely on Windows or Linux."""
    if not process:
        return
        
    pid = process.pid
    try:
        if IS_WINDOWS:
            subprocess.run(f"taskkill /F /T /PID {pid}", shell=True)
        else:
            # Linux: Kill process group
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception as e:
        logger.error(f" Failed to kill process {pid}: {e}")
        # Fallback
        try:
            process.terminate()
        except:
            pass

def main():
    logger.info(f" Supervisor Initialized ({platform.system()}). Starting Infinite Loop...")
    
    bot_process = None
    
    while True:
        try:
            # 1. Start Bot
            if bot_process is None:
                cmd = f"python {BOT_SCRIPT} --capture"
                
                # CHECK: Do we have a trained brain yet?
                # If no model exists, force DRY RUN to be safe (Bootsrap Phase)
                if not os.path.exists("models/direction_classifier.joblib"):
                    logger.info(" No AI Brain found. Forcing DRY-RUN for data collection phase.")
                    if "--dry-run" not in sys.argv:
                        cmd += " --dry-run"
                
                # Pass through user arguments (e.g. --dry-run)
                extra_args = " ".join(sys.argv[1:])
                if extra_args:
                    cmd += f" {extra_args}"
                
                # On Linux, we want to start the bot in a new process group so we can kill the whole tree later
                if not IS_WINDOWS:
                    bot_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
                else:
                    bot_process = run_process(cmd, wait=False)
            
            # 2. Wait for Cycle (Minus Lead Time)
            wait_time = max(0, CYCLE_TIME_SECONDS - LEAD_TIME_SECONDS)
            logger.info(f" Bot running. Waiting {wait_time/3600:.2f} hours before optimization...")
            time.sleep(wait_time)
            
            # 3. Parallel Optimization (Bot is still running!)
            logger.info(" Starting Parallel Optimization (Bot still active)...")
            
            latest_file = get_latest_capture_file()
            if latest_file:
                # Copy to temp file to avoid locking issues
                temp_file = os.path.join(DATA_DIR, "temp_optimize.jsonl")
                try:
                    shutil.copy(latest_file, temp_file)
                    logger.info(f" Snapshot taken: {latest_file} -> {temp_file}")
                    
                    # Optimize on snapshot (Low Priority)
                    logger.info(" Running Optimizer on snapshot...")
                    run_process(f"python {OPTIMIZER_SCRIPT} {temp_file}", wait=True, low_priority=True)
                    
                    # Train on snapshot (Low Priority)
                    logger.info(" Training AI Model on snapshot...")
                    run_process(f"python {TRAINER_SCRIPT} {temp_file}", wait=True, low_priority=True)
                    
                    # Cleanup temp
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    logger.error(f" Parallel Optimization Failed: {e}")
            else:
                logger.warning(" No data found for optimization.")

            # 4. Stop Bot
            logger.info(" Cycle complete. Restarting bot to apply new brain...")
            if bot_process:
                kill_process(bot_process)
                bot_process = None
                time.sleep(5) 
            
            # 5. Cleanup Old Data
            cleanup_old_files()
            
            logger.info(" Restarting loop...")
            logger.info("")
            
        except KeyboardInterrupt:
            logger.info(" Supervisor stopped by user.")
            if bot_process:
                kill_process(bot_process)
            break
        except Exception as e:
            logger.error(f" Supervisor Error: {e}")
            time.sleep(60) 

if __name__ == "__main__":
    main()
