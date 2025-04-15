import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def plot_pre_post_tms_comparison(pre_mean, pre_std, post_mean, post_std, save_path='tms_comparison.pdf'):
    """Generate and save a bar chart comparing Pre- and Post-TMS errors."""
    # Prepare data
    df = pd.DataFrame({
        'Condition': ['Pre-TMS', 'Post-TMS'],
        'Mean Error': [pre_mean, post_mean],
        'Std Dev': [pre_std, post_std]
    })

    # Plot setup
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=df,
        x='Condition',
        y='Mean Error',
        hue='Condition',
        palette='muted',
        dodge=False,
        legend=False
    )

    # Add error bars manually
    for i, row in df.iterrows():
        ax.errorbar(
            x=i,
            y=row['Mean Error'],
            yerr=row['Std Dev'],
            fmt='none',
            ecolor='black',
            capsize=8,
            capthick=2,
            linewidth=1.5
        )

    # Annotate bars
    for i, row in df.iterrows():
        ax.text(i, row['Mean Error'] + row['Std Dev'] + 0.3, f"{row['Mean Error']:.2f}", 
                ha='center', va='bottom', fontsize=12)

    # Labeling
    plt.title('Accumulation Error Before and After TMS', fontsize=14)
    plt.ylabel('Accumulation Error', fontsize=12)
    plt.xlabel('')
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(save_path, format='pdf')
    plt.close()


def list_txt_file_paths(directory):
    """Return a list of full paths to all .txt files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]


def load_eeg_trials(filepath):
    trials = []
    current_trial = None
    reading_data = False
    current_condition = None  # 'rest' or 'task'

    def finalize_trial(trial, label, condition):
        if trial['time']:  # only add non-empty trials
            df = pd.DataFrame(trial)
            df.attrs['label'] = label
            df.attrs['condition'] = condition
            trials.append(df)

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            # Detect event marker
            if parts[0] == parts[1] == parts[2]:
                try:
                    code = int(parts[0])
                    if code == 7691:
                        current_trial = {'time': [], 'LH': [], 'RH': []}
                        current_condition = 'rest'
                        reading_data = True
                    elif code == 7701:
                        current_trial = {'time': [], 'LH': [], 'RH': []}
                        current_condition = 'task'
                        reading_data = True
                    elif code in [7692, 7702]:  # Miss
                        if current_trial:
                            finalize_trial(current_trial, 'miss', current_condition)
                            current_trial = None
                        reading_data = False
                    elif code in [7693, 7703]:  # Hit
                        if current_trial:
                            finalize_trial(current_trial, 'hit', current_condition)
                            current_trial = None
                        reading_data = False
                except ValueError:
                    continue
            else:
                if reading_data and current_trial:
                    try:
                        t, lh, rh = map(float, parts)
                        current_trial['time'].append(t)
                        current_trial['LH'].append(lh)
                        current_trial['RH'].append(rh)
                    except ValueError:
                        continue

    return trials


def compute_accumulation_error(trial_df):
    condition = trial_df.attrs.get('condition', None)
    if condition == 'rest':
        error = np.abs(trial_df['LH'] - 65).sum()
    elif condition == 'task':
        error = np.abs(trial_df['RH'] - 65).sum()
    else:
        raise ValueError("Unknown condition: must be 'rest' or 'task'")
    return error


# Example usage
if __name__ == '__main__':
    for subject in ['207', '208', '206']:
        path = f'/home/ss227376/NeuralEngineering/Project/data/Group2/Subject_{subject}_FES_Online/Subject_{subject}_Session_002_FES_Online_Visual/'
        txt_file_names = list_txt_file_paths(path)
        print(f"Total .txt files found: {len(txt_file_names)}")
        print(txt_file_names)
        pre_tms_file_paths = txt_file_names[:3]
        post_tms_file_paths = txt_file_names[3:]
        # path = '/home/ss227376/NeuralEngineering/Project/data/Group2/Subject_207_FES_Online/Subject_207_Session_002_FES_Online_Visual/Subject_207_FES_Online__feedback_n_s002_r001_2025_03_26_090749_probabilities_log.txt'  # Replace with your actual file path
        path = pre_tms_file_paths[1]  # Replace with your actual file path
        trials = load_eeg_trials(path)

        print(f"Total trials loaded: {len(trials)}")
        for i, trial in enumerate(trials[:3]):
            print(f"\nTrial {i+1} ({trial.attrs['condition']} | {trial.attrs['label']}):")
            print(trial.head())
        
        pre_tms_errors = []
        post_tms_errors = []
        for file_path in pre_tms_file_paths:
            trials = load_eeg_trials(file_path)
            pre_tms_errors.extend([compute_accumulation_error(trial) for trial in trials])
        for file_path in post_tms_file_paths:   
            trials = load_eeg_trials(file_path)
            post_tms_errors.extend([compute_accumulation_error(trial) for trial in trials])
        print(f"\nPre-TMS Errors: {np.mean(pre_tms_errors)}")
        print(f"\nPre-TMS Errors std: {np.std(pre_tms_errors)}")
        print(f"\nPost-TMS Errors: {np.mean(post_tms_errors)}")
        print(f"\nPost-TMS Errors std: {np.std(post_tms_errors)}")
        plot_pre_post_tms_comparison(
            np.mean(pre_tms_errors), np.std(pre_tms_errors),
            np.mean(post_tms_errors), np.std(post_tms_errors),
            save_path = 'subject_' + subject +'_tms_comparison.pdf'
        )
        
        # for i, trial in enumerate(trials):
        #     error = compute_accumulation_error(trial)
        #     print(f"Trial {i+1}: label={trial.attrs['label']}, condition={trial.attrs['condition']}, error={error:.2f}")
