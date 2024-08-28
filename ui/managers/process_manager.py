import subprocess
import multiprocessing
import threading
import queue
import asyncio

class Process:
    def __init__(self, console, command, working_directory):
        self.console = console
        self.command = command
        self.working_directory = working_directory

class ProcessManager:
    def __init__(self):
        self.process = None
        self.multproc = None
        self.console = None
        self.running = False

        self.process_queue = queue.Queue()

        pass
    
    def create_process(self, command, working_directory, stdout_queue, status_queue):
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=False
        )

        def enqueue_output(pipe, queue):
            for line in iter(pipe.readline, ''):
                queue.put(line)
            pipe.close()

        stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stdout_queue))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        status_queue.put("STARTED")
        process.wait()
        status_queue.put("COMPLETED")

        return process

    def start_process(self, console, command, working_directory):
        if self.running:
            print("Process already running", self.process_queue.qsize())    
            #self.process_queue.put(Process(console, command, working_directory))
            return

        self.running = True

        self.console = console
        self.console.kill_button.configure(command=self.kill_process)

        self.console.console_window.clear()
        self.console.show_progress()

        stdout_queue = multiprocessing.Queue()
        status_queue = multiprocessing.Queue()

        def process_wrapper():
            self.process = self.create_process(command, working_directory, stdout_queue, status_queue)

        self.multproc = multiprocessing.Process(target=process_wrapper)
        self.multproc.start()

        self.console.after(100, self.poll_process_status, status_queue)
        self.console.after(100, self.update_console, stdout_queue)

    def poll_process_status(self, status_queue):
        try:
            status = status_queue.get_nowait()
            if status == "STARTED":
                self.running = True 
            if status == "COMPLETED":
                self.running = False
                self.console.progressbar.configure(progress_color="green")
                self.console.console_window.write("Process completed\n")
                self.process_completed()
        except queue.Empty:
            pass

        if self.running:
            self.console.after(100, self.poll_process_status, status_queue)

    def update_console(self, stdout_queue):
        if not self.console:
            return

        try:
            line = stdout_queue.get_nowait()

            self.console.console_window.write(line)
            self.console.display_progress_info(line)
        except queue.Empty:
            pass

        if self.running:
            self.console.after(100, self.update_console, stdout_queue)

    def process_completed(self):
        self.running = False
        self.process = None
        self.multproc = None
        if self.console:
            #self.console.hide_progress()
            #self.console.console_window.clear()
            self.console = None

        """try:
            next_process = self.process_queue.get()
            self.start_process(next_process.console, next_process.command, next_process.working_directory)
        except queue.Empty:
            pass"""   

    def kill_process(self):
        #with self.process_queue.mutex:
        #    self.process_queue.queue.clear()

        try:
            if self.process:
                self.process.kill()
            if self.multproc:
                self.multproc.terminate()

            if self.console:
                self.console.progressbar.configure(progress_color="orange")
                self.console.console_window.write("Process terminated\n")

            self.process_completed()

        except Exception as e:
            print(f"Error terminating process: {e}")