DATASET = {
    'uag_oops': {
        'file': '/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/oops_uag_paper_version.json',
        'values': ['start_time','end_time','description'],
        'video_dir':'/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/video'
    }
}
vlm = [
    'blip2'
]


class Config:
    def __init__(self):
        self.DATASET = DATASET
        self.vlm = vlm
