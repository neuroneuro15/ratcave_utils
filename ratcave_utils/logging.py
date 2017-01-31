import csv
import ratcave as rc
import motive
import time


class Logger(rc.utils.Observer):

    columns=('MotiveTime', 'Time', 'Name', 'Visible', 'x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z', 'glob_x', 'glob_y', 'glob_z')

    def __init__(self, fname, **kwargs):
        """Creates a CSV Logging object that writes whenever registered ratcave Observable objects change."""
        super(Logger, self).__init__(**kwargs)
        self.fname = fname
        self.writer = None
        self.f = None

    def open(self):
        self.f = open(self.fname, 'wb')
        self.writer = csv.DictWriter(self.f, fieldnames=self.columns)
        self.writer.writeheader()

    def close(self):
        self.f.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def on_change(self):
        super(Logger, self).on_change()

        if self._changed_observables:
            motive_tt = motive.frame_time_stamp()
            tt = time.clock()
            for obs in self._changed_observables:
                line = {}
                line['MotiveTime'] = motive_tt
                line['Time'] = tt
                line['Name'] = obs.name
                line['Visible'] = obs.visible
                line['x'], line['y'], line['z'] = obs.position.xyz
                rot_euler = obs.rotation.to_euler(units='rad')
                line['rot_x'], line['rot_y'], line['rot_z'] = rot_euler.xyz
                rot_quat = obs.rotation.to_quaternion()
                line['quat_w'], line['quat_x'], line['quat_y'], line['quat_z'] = rot_quat.wxyz
                line['glob_x'], line['glob_y'], line['glob_z'] = obs.position_global
                self.writer.writerow(line)
