import os

from cv2 import cv2

videos_dir = '../dataset/video/ours'
frames_dir = '../dataset/frames/ours'
fps = 10

def extract_frames(video_path, fps=10):
    """
        Extrage cadrele dintr-un fișier video la un anumit FPS.
        :param video_path: Calea către fișierul video.
        :param fps: Numărul de cadre pe secundă de extras.
        :return: O listă de cadre.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps == 0: #din 10 frames se ia 1
            frames.append(frame)
        count += 1
    cap.release()
    return frames

if __name__ == '__main__':

    for video_file in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, video_file)
        frames = extract_frames(video_path, fps=fps)
        video_name = os.path.splitext(video_file)[0]
        video_frames_dir = os.path.join(frames_dir)
        os.makedirs(video_frames_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_file = f"{video_name}{i}.jpg"
            frame_path = os.path.join(video_frames_dir, frame_file)
            cv2.imwrite(frame_path, frame)
