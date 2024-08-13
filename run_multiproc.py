import os
import pandas as pd
import numpy as np
import time
import concurrent.futures

def get_distance_velocity(df, side = 'right'):


    df[f'{side}_distance_x'] = df[f'{side} finger']['x'] - df['nosetip']['x']
    df[f'{side}_distance_y'] = df[f'{side} finger']['y'] - df['nosetip']['y']

    df[f'{side}_velocity_x'] = df[f'{side} finger']['x'].diff()
    df[f'{side}_velocity_y'] = df[f'{side} finger']['y'].diff()

    df[f'{side}_distance'] = np.sqrt(df[f'{side}_distance_x']**2 + df[f'{side}_distance_y']**2)
    df[f'{side}_velocity'] = np.sqrt(df[f'{side}_velocity_x']**2 + df[f'{side}_velocity_y']**2)

    df[f'{side}_distance_pelllet_x'] = df[f'{side} finger']['x'] - df['pellet']['x']
    df[f'{side}_distance_pelllet_y'] = df[f'{side} finger']['y'] - df['pellet']['y']

    df[f'{side}_distance_pelllet'] = np.sqrt(df[f'{side}_distance_pelllet_x']**2 + df[f'{side}_distance_pelllet_y']**2)
    return df

def prepair_data(df, smooth = 5):

    df[f'right finger']['x'] = df[f'right finger']['x'].rolling(smooth).mean()
    df[f'right finger']['y'] = df[f'right finger']['y'].rolling(smooth).mean()
    df[f'left finger']['x'] = df[f'left finger']['x'].rolling(smooth).mean()
    df[f'left finger']['y'] = df[f'left finger']['y'].rolling(smooth).mean()
    df[f'nosetip']['x'] = df[f'nosetip']['x'].rolling(smooth).mean()
    df[f'nosetip']['y'] = df[f'nosetip']['y'].rolling(smooth).mean()
    df[f'pellet']['x'] = df[f'pellet']['x'].rolling(smooth).mean()
    df[f'pellet']['y'] = df[f'pellet']['y'].rolling(smooth).mean()
    df = get_distance_velocity(df, 'right')
    return df

def extract_episodes(df, side='right', start_threshold=0, end_threshold_min=-30, end_threshold_max=0):
    episodes = []
    episode_start = None
    backward = False

    n = len(df)
    # get cpu thread count
    process_n = os.cpu_count()
    
    target_index_list = np.linspace(0, n, process_n+1).astype(int)
    print(target_index_list)

    target_index_shift = 1000

    # given [0,1,2] get [[0,1],[1,2]]
    target_index_list = [[target_index_list[i]-target_index_shift, target_index_list[i+1]] for i in range(len(target_index_list)-1)]
    target_index_list[0][0] = 0

    print(target_index_list)

    # run in parallel

    for target_index in target_index_list:
        print(target_index)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(extract_episodes_each_loop, df.iloc[target_index[0]:target_index[1]], side, start_threshold, end_threshold_min, end_threshold_max, shifter=target_index[0]) for target_index in target_index_list]
        for result in concurrent.futures.as_completed(results):
            episodes += result.result()

    #sort the episodes
    episodes = sorted(episodes, key=lambda x: x[0])

    # save episodes as temp.csv
    with open('temp.csv', 'w') as f:
        for episode in episodes:
            f.write(f'{episode[0]},{episode[1]}\n')

    episode_dfs = pd.DataFrame(columns=['start', 'end'], data=episodes)
    return episode_dfs

def extract_episodes_each_loop(df, side='right', start_threshold=0, end_threshold_min=-30, end_threshold_max=0, shifter=0):
    episodes = []
    episode_start = None
    backward = False
    n = len(df)

    process_id = os.getpid()

    for i in range(1, n):
        tic = time.time()
        # Start of a new episode: velocity changes from non-positive to positive
        # if episode_start is not None and backward:
        #     print(df[f'bodyparts']['coords'].iloc[i], df[f'{side}_velocity_x'].iloc[i], backward, df[f'{side}_distance_x'].iloc[i])
        #     print( episode_start is not None,  df[f'{side}_velocity_x'].iloc[i] < end_threshold_max , df[f'{side}_velocity_x'].iloc[i] > end_threshold_min,  backward,  df[f'{side}_distance_x'].iloc[i] < 0)
        if (df[f'{side}_velocity_x'].iloc[i] > start_threshold and (episode_start is None) and (not backward)):
            episode_start = i

        elif df[f'{side}_velocity_x'].iloc[i] < 0 and episode_start is not None and not backward:

            backward = True

        # End of an episode: velocity changes from positive to non-positive
        elif episode_start is not None and df[f'{side}_velocity_x'].iloc[i] < end_threshold_max and df[f'{side}_velocity_x'].iloc[i] > end_threshold_min and backward and df[f'{side}_distance_x'].iloc[i] < 0:
            episodes.append((episode_start, i))

            episode_start = None
            backward = False

        if i % 1000 == 0:
            toc = time.time()
            eta_sec = (toc-tic)*(n-i)
            eta_min = int(eta_sec//60)
            eta_sec = int(eta_sec%60)

            print(f'P{process_id} - {i/n*100} % - ETA: {eta_min}:{eta_sec:02d}')

    # add shifter to the episode
    episodes = [(episode[0]+shifter, episode[1]+shifter) for episode in episodes]

    return episodes


def clean_episodes(df, episode_dfs, side='right', frame_threshold=30, finger_likelihood = 0.5, nosetip_likelihood = 0.5):
    clean_episode_df = pd.DataFrame(columns=['start', 'end', 'max_velocity', 'mean_velocity', 'min_pellet', 'min_frame'])

    for i, row in episode_dfs.iterrows():
        start = row['start']
        end = row['end']

        if df.iloc[start:end][f'{side} finger']['likelihood'].max() > finger_likelihood and df.iloc[start:end]['nosetip']['likelihood'].max() > nosetip_likelihood:
            if end - start > frame_threshold:
                max_velocity = df.iloc[start:end][f'{side}_velocity'].max()
                mean_velocity = df.iloc[start:end][f'{side}_velocity'].abs().mean()
                min_pellet = df.iloc[start:end][f'{side}_distance_pelllet'].min()
                min_pellet_loc = df.iloc[start:end][f'{side}_distance_pelllet'].idxmin()
                min_frame = df.iloc[min_pellet_loc]['bodyparts']['coords']
                clean_episode_df = pd.concat([clean_episode_df, pd.DataFrame([[start, end, max_velocity, mean_velocity, min_pellet, min_frame]], columns=clean_episode_df.columns)], ignore_index=True)

    return clean_episode_df

# given episode dataframes, combine the overlap episodes into one if start of the next episode is within the previous episode
def combine_overlap(episode_dfs):
    episode_dfs = episode_dfs.sort_values(by='start')

    for i in range(len(episode_dfs)-1):
        if episode_dfs.loc[i,'end'] > episode_dfs.loc[i+1,'start']:
            episode_dfs.loc[i+1,'start'] = -1
            episode_dfs.loc[i,'end'] = episode_dfs.loc[i+1,'end']

    # dros the rows with start = -1
    episode_dfs = episode_dfs[episode_dfs['start'] != -1]

    return episode_dfs

# from episode_dfs, remove duplicate episodes, dataframe already sorted by start
def remove_duplicate(episode_dfs):
    episode_dfs = episode_dfs.reset_index(drop=True)
    for i in range(len(episode_dfs)-1):
        if episode_dfs.loc[i,'end'] == episode_dfs.loc[i+1,'end']:
            episode_dfs.loc[i+1,'start'] = -1

    # dros the rows with start = -1
    episode_dfs = episode_dfs[episode_dfs['start'] != -1]

    return episode_dfs


def main():

    # Path Input
    input_dir = './input_fromdlc'
    output_dir = './output_episodes'
    side = 'right'


    #input_filename = "baa937766dcb2bd1_scaledDLC_mobnet_100_y79Jun20shuffle1_10000.csv"

    filename_list = os.listdir(input_dir)
    print(filename_list)

    for input_filename in filename_list:
        if input_filename.split('.')[-1] == 'csv':    
            input_filepath = os.path.join(input_dir, input_filename)
            output_filename = input_filename.split('.')[0] + '_episodes.csv'
            output_filepath = os.path.join(output_dir, output_filename)

            df = pd.read_csv(input_filepath, skiprows=1, header = [0,1])
            df = prepair_data(df)

            episode_dfs = extract_episodes(df, side)

            #episode_dfs = combine_overlap(episode_dfs)
            #print(f'Combined {len(episode_dfs)} episodes')

            clean_episode_df = clean_episodes(df, episode_dfs, side)
            clean_episode_df = remove_duplicate(clean_episode_df)
            clean_episode_df.to_csv(output_filepath, index=False)

            print(f'Output file saved at {output_filepath}')
    

if __name__ == '__main__':
    main()