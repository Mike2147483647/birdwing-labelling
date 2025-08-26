import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# create path to dir of log locations
birdwinglabel_dir = Path(__file__).parent.parent / 'src' / 'birdwinglabel'
MLP_dir = birdwinglabel_dir / 'MLP'
ET_dir = birdwinglabel_dir / 'EncoderOnlyTransformers'


# read in logs
def read_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines

# extract values as dataframe by keys ('Avg loss', 'Accuracy per entry')
def extract_metrics_from_log(lines):
    losses = []
    accuracies_frame = []
    accuracies_entry = []
    for line in lines:
        loss_match = re.search(r'Avg loss:\s*([0-9.]+)', line)
        acc_frame_match = re.search(r'Accuracy per frame:\s*([0-9.]+)', line)
        acc_entry_match = re.search(r'Accuracy per entry:\s*([0-9.]+)', line)
        if loss_match:
            losses.append(float(loss_match.group(1)))
        if acc_frame_match:
            accuracies_frame.append(float(acc_frame_match.group(1)))
        if acc_entry_match:
            accuracies_entry.append(float(acc_entry_match.group(1)))
    return pd.DataFrame({
        'Avg loss': losses,
        'Accuracy per frame': accuracies_frame,
        'Accuracy per entry': accuracies_entry
    })

# wrapper of extract and read
def extract_dataframe_from_log(filepath):
    return extract_metrics_from_log(read_log_file(filepath))

# main plotter
class LogMetricsPlotter_labels:
    def __init__(self, df_list):
        # df_list: list of (name, dataframe) tuples
        # eg: plotter = LogMetricsPlotter([('name1', df1), ('name2', df2)])
        self.log_dataframes = {name: df for name, df in df_list}

    def __getitem__(self, key):
        return self.log_dataframes[key]

    def plot_loss(self, title=None, figsize=(8, 5), xlabel='Epoch', ylabel='Avg Loss', save_path=None,
                  xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Avg loss'], label=name)
            plt.xticks(range(0, len(df['Avg loss']), xtick_step), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title:
            plt.title(title, fontsize=fontsize + 2)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize - 2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_accuracy_per_frame(self, title=None, figsize=(8, 5), xlabel='Epoch',
                                ylabel='Accuracy per frame', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Accuracy per frame'], label=name)
            plt.xticks(range(0, len(df['Accuracy per frame']), xtick_step), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title:
            plt.title(title, fontsize=fontsize + 2)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize - 2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_accuracy_per_entry(self, title=None, figsize=(8, 5), xlabel='Epoch',
                                ylabel='Accuracy per Entry', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Accuracy per entry'], label=name)
            plt.xticks(range(0, len(df['Accuracy per entry']), xtick_step), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title:
            plt.title(title, fontsize=fontsize + 2)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize - 2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
if __name__ == '__main__':
    current_dir = Path(__file__).parent

    # MLP_8m_df = extract_metrics_from_log(read_log_file(MLP_dir / 'MLP_8m_train_log.txt'))
    # MLP_wrong_loss_df = extract_metrics_from_log(read_log_file(MLP_dir / 'wrong_loss_train_log.txt'))
    # ET_8m_df = extract_metrics_from_log(read_log_file(ET_dir / 'ET_simple_train_log.txt'))
    # MLP_plotter = LogMetricsPlotter_labels([('MLP', MLP_8m_df), ('wrong loss', MLP_wrong_loss_df), ('ET', ET_8m_df)])
    # MLP_plotter.plot_accuracy_per_entry(save_path= current_dir / '8 markers' / 'accuracy per entry.pdf')
    # MLP_plotter.plot_loss(save_path=current_dir / '8 markers' / 'test loss.pdf')
    # MLP_plotter.plot_accuracy_per_frame(save_path=current_dir / '8 markers' / 'accuracy per frame.pdf')

    ET_32m_df = extract_dataframe_from_log(ET_dir / 'ET_hard_train_log.txt')
    ET_32m_more_data_df = extract_dataframe_from_log(ET_dir / 'ET_32m_more_data_train_log.txt')
    ET_plotter = LogMetricsPlotter_labels([('ET', ET_32m_df), ('ET+', ET_32m_more_data_df)])
    ET_plotter.plot_loss(save_path=current_dir / 'ET hard' / 'test loss.pdf')
    ET_plotter.plot_accuracy_per_frame(save_path=current_dir / 'ET hard' / 'accuracy per frame.pdf')









