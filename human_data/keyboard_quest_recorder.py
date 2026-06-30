import datetime
import os
from human_data.quest_recorder import QuestRecorder


class KeyboardQuestRecorder(QuestRecorder):
    """
    One left-middle-pinch at session start puts Quest into streaming mode (sends "Start").
    After that, all further Quest gestures are ignored and recording is controlled entirely
    by keyboard via start_recording_manual() / stop_recording_manual().
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
        status, xrhand, head_pose, timestamp = super().receive(verbose=verbose)
        # "Start" from Quest (first pinch) sets quest_recording=True via super() — keep that
        # side effect so hand data starts flowing, but return "Wait" to main loop.
        # Any further Quest gestures (Ensure/Save/Cancel) would flip quest_recording=False;
        # re-assert True to keep the stream alive.
        if status in ("Ensure", "Save", "Cancel", "Wait-Ensure"):
            self.quest_recording = True
            return "Wait", None, None, timestamp
        if status == "Data":
            return "Data", xrhand, head_pose, timestamp
        return "Wait", None, None, timestamp
