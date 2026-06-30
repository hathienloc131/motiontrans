import datetime
import os
import time
import numpy as np
from queue import Empty
from human_data.quest_recorder import QuestRecorder
from human_data.models import XRHand, Transform


class KeyboardQuestRecorder(QuestRecorder):
    """
    Bypass all Quest gesture state machine. Read raw packets directly from the queue and
    extract hand data regardless of whatever status prefix Quest prepends (e.g. "Wait...").
    Recording state is controlled entirely by keyboard via start/stop_recording_manual().
    """

    def start_recording_manual(self):
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.data_dir = os.path.join(self.output_dir, formatted_time)
        os.mkdir(self.data_dir)
        self.quest_recording = True
        return self.data_dir

    def stop_recording_manual(self):
        self.quest_recording = False

    def receive(self, verbose=False):
        # Read raw bytes directly — bypass parent's state machine entirely.
        while True:
            try:
                data = self.data_queue.get(block=False)
                break
            except KeyboardInterrupt:
                self.close()
                exit(0)
            except Empty:
                pass

        timestamp = time.time()
        data_string = data.decode()

        if verbose:
            print(f"[KeyboardQuestRecorder] raw packet: {data_string[:80]}")

        # If the packet contains hand data (regardless of "Wait" or any other prefix),
        # parse and return it. Quest embeds LHand/RHand in every packet.
        if "LHand:" in data_string and "RHand:" in data_string:
            if not self.quest_recording:
                return "Wait", None, None, timestamp
            try:
                st = data_string.find("LHand:") + len("LHand:")
                ed = data_string.find("RHand:")
                left_hand = XRHand(data_string[st:ed])
                right_hand = XRHand(data_string[ed + len("RHand:"):])

                # head data sits between offset 11 and "LHand:" — keep same format as parent
                head_str = data_string[11:data_string.find("LHand:")]
                head_data_list = [float(v) for v in head_str.split(",") if v.strip()]
                head_tf = np.array(head_data_list[:7])

                self.compute_rel_transform_for_hand(left_hand)
                self.compute_rel_transform_for_hand(right_hand)
                rel_head_pos, rel_head_rot = self.compute_rel_transform(head_tf)
                head_pose = Transform()
                head_pose.set_pose(np.concatenate([rel_head_pos, rel_head_rot]))
                return "Data", (left_hand, right_hand), head_pose, timestamp
            except Exception as e:
                if verbose:
                    print(f"[KeyboardQuestRecorder] parse error: {e}")
                return "Wait", None, None, timestamp

        return "Wait", None, None, timestamp
