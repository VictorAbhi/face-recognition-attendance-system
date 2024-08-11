import os
import csv
from datetime import datetime
import time


class AttendanceLogger:
    def __init__(self, attendance_dir='Attendance', col_names=['NAME', 'TIME']):
        self.attendance_dir = attendance_dir
        self.col_names = col_names

        # Create the attendance directory if it doesn't exist
        if not os.path.exists(self.attendance_dir):
            os.makedirs(self.attendance_dir)

    def log_attendance(self, name):
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        attendance = [name, timestamp]
        attendance_file = os.path.join(self.attendance_dir, f'Attendance_{date}.csv')

        if os.path.isfile(attendance_file):
            with open(attendance_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.col_names)
                writer.writerow(attendance)
