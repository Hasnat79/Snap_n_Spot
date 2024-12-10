import os
import sys
root = os.path.join(os.path.dirname(__file__), '..')

DATASETS={
    'uag_oops': {
        'feature_path': '/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/blip_2_features',
        # 'stride': 20,
        'stride': 20,
        # 'max_stride_factor': 0.5,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                # 'annotation_file': 'dataset/charades-sta/charades_test.json',
                'annotation_file': '/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/uag_oops_charades_format.json',
                'pad_sec': 0.0,
            }
        }
    },
    'charades': {
        'feature_path': '/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/charades-sta/features',
        'video_dir':'/scratch/user/hasnat.md.abdullah/uag/data/charades_sta/Charades_v1_480',
        # 'stride': 20,
        'stride': 20,
        # 'max_stride_factor': 0.5,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                # 'annotation_file': 'dataset/charades-sta/charades_test.json',
                'annotation_file': '/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            }
        }
    }
}
