class Statistics:
    def __init__(self, filename):
        self.frame_count = 0
        self.face_count = 0
        self.sleep_count = 0
        self.filename = filename

    def increment_frame_count(self):
        self.frame_count += 1

    def increment_face_count(self):
        self.face_count += 1

    def increment_sleep_count(self):
        self.sleep_count += 1

    def generate_statistics(self):
        statistics = {
            'frame_count': self.frame_count,
            'face_count': self.face_count,
            'sleep_count': self.sleep_count
        }

        with open(self.filename, 'w') as file:
            for key, value in statistics.items():
                file.write(f'{key}: {value}\n')

        print(f'Statistics written to {self.filename}')
