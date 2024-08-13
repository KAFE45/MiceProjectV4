import cv2
import os
import pandas as pd

class Session():

    DEFAULT_VIDEO_EXTENTION = ".MOV"        # Video file

    def __init__(self, session_id: str, episode_id: str):
        self.__session_id = session_id
        self.__episode_id = episode_id
        self.set_videofile_dir()

    def get_default_session_dir(self):
        return os.path.join(os.getcwd(), 'Dataset')

    def set_videofile_dir(self, path=None):
        if path:
            self.__video_dir = path
        else:
            self.__video_dir = self.get_default_session_dir()

    def get_videofile_path(self):
        return os.path.join(self.__video_dir, f"{self.__session_id}{self.DEFAULT_VIDEO_EXTENTION}")
    
    def get_episodefile_path(self):
        self.__episode_dir = os.path.join(os.getcwd(), 'output_episodes')
        return os.path.join(self.__episode_dir, f"{self.__episode_id}{'.csv'}")


def render_ui_frame_id(frame, frame_id, frame_total):
    if is_eps == False:
        print(f'{frame_total} : {frame_id}')
    else:
        cv2.rectangle(frame, (10, 5), (400, 35), (0,0,0), -1)
        cv2.putText(frame, f'{frame_total} : {frame_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

   
if __name__ == '__main__':
    session_id = "4ed06efa3648f952_scaled" 
    episode_id = "4ed06efa3648f952_scaledDLC_mobnet_100_y79Jun20shuffle1_10000_episodes"
    session_handler = Session(session_id,episode_id)
    is_eps = True
    if is_eps:
        episode_path = session_handler.get_episodefile_path()
        df = pd.read_csv(episode_path, usecols=['start', 'end'])
        df_list = df.values.tolist()
        print(df_list[0][0])



    # Load the video file
    video_path = session_handler.get_videofile_path()
    capture_handler = cv2.VideoCapture(video_path)
    if not capture_handler.isOpened():
        print(f"Error: Unable to open the video file {video_path}")
        exit()

    # Get the total frame of the video%
    frame_total = int(capture_handler.get(cv2.CAP_PROP_FRAME_COUNT))


    # Define the window name, also use it as tag for using in CV
    window_name = f"{session_id} - mICE Video Marker"

    # Create a window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_id_current = 0
    saved_frame = None
    flag_pause = True
    is_on_seeking = False
    cnt_episode_frame = 0

    while True:
        # Check if the video is paused, if so, then keep playing the current frame
        if is_on_seeking:
            if flag_is_set_marker_founded:
                is_on_seeking = False
                flag_is_set_marker_founded = False
                flag_pause = True
                saved_frame = None
                capture_handler.set(cv2.CAP_PROP_POS_FRAMES, frame_id_current)

        if flag_pause:
            if saved_frame is None:
                ret, saved_frame = capture_handler.read()
            frame = saved_frame.copy()
        else:
            ret, frame = capture_handler.read()

        frame_id_current = int(capture_handler.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Render frame info
        render_ui_frame_id(frame, frame_id_current, frame_total)
        
        # frameS = cv2.resize(frame,(640,360))
        # cv2.resizeWindow('jpg', 640, 360)
        # Display the frame 
        cv2.imshow(window_name, frame)

        # Wait for a key press, q to exit, space to start/stop the video (keep playing current frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'),ord('Q')]:
            exit()

        elif key in [ord('R'),ord('r'),ord('d'),ord('D'),ord(' '),ord('Z'),ord('z'),ord('x'),ord('v'),ord('b'),ord('n'),ord ('m')]:
            if key == ord(' '):
                is_on_seeking = False
                if flag_pause:
                    flag_pause = False
                    saved_frame = None
                else:
                    flag_pause = True
                frame_id_current = int(capture_handler.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if is_eps:
                if key==ord('r') or key==ord('R'):
                    cnt_episode_frame -=1
                    frame_id_current = df_list[cnt_episode_frame][0] 
                if key==ord('d') or key==ord('D'):
                    cnt_episode_frame +=1
                    # while df_list[cnt_episode_frame][0] < frame_id_current:
                    #     cnt_episode_frame+=1
                    frame_id_current = df_list[cnt_episode_frame][0] 
                

            if key == ord('Z'):
                frame_id_current = 0

            if key == ord('z'):
                frame_id_current -= 240

            if key == ord('x'):
                frame_id_current -= 1

            if key == ord('v'):
                frame_id_current += 1

            if key == ord('b'):
                frame_id_current += 240

            if key == ord('n'):
                frame_id_current += 120

            if key == ord('m'):
                frame_id_current += 500
            capture_handler.set(cv2.CAP_PROP_POS_FRAMES, frame_id_current)
            saved_frame = None
