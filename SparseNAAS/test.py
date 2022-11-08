
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}

'''


import os
import tqdm
import time

files = ["a","b",'c','d','e']

outer = tqdm.tqdm(total=len(files), desc='Files', position=0)

# Three logging bars for filename, video stats, and frame name
# important is that the bar_format='{desc}' so that the bar
# shows only the description
file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
video_log = tqdm.tqdm(total=0, position=3, bar_format='{desc}')
frame_log = tqdm.tqdm(total=0, position=4, bar_format='{desc}')

counter = 0
for filename in files:
    video =  [1,2,3]
    inner = tqdm.tqdm(total=len(video), desc='Frames', position=2)

    # It is important to use set_description_str as the set_description
    # method adds a colon at the end of the line
    file_log.set_description_str(f'Current file: {filename}')
    video_log.set_description_str(f'{len(video)} frames of {video[0]}x{video[0]} pixels')
    for frame in video:
        time.sleep(1)
        frame_log.set_description_str(f'{counter:08d}.png saved')
        inner.update(1)
        counter += 1
    outer.update(1)
'''
