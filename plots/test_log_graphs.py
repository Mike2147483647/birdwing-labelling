import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# create path to dir of log locations
birdwinglabel_dir = Path(__file__).parent.parent / 'src' / 'birdwinglabel'
MLP_dir = birdwinglabel_dir / 'MLP'
ET_dir = birdwinglabel_dir / 'EncoderOnlyTransformers'
AT_dir = birdwinglabel_dir / 'EncDecTransformers'

# read in logs
def read_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines

# extract values as dataframe by keys ('Avg loss', 'Accuracy per entry')
def extract_metrics_from_log(lines):
    metrics = {
        'Avg loss': [],
        'Accuracy per frame': [],
        'Accuracy per entry': [],
        'Test loss avg over frame': [],
        'Frames <5% error': [],
        'Frames <10% error': [],
        'Frames <20% error': []
    }
    patterns = {
        'Avg loss': r'Avg loss:\s*([0-9.]+)',
        'Accuracy per frame': r'Accuracy per frame:\s*([0-9.]+)',
        'Accuracy per entry': r'Accuracy per entry:\s*([0-9.]+)',
        'Test loss avg over frame': r'test loss avg over frame:\s*([0-9.]+)',
        'Frames <5% error': r'frames that has <5% error:\s*([0-9.]+)',
        'Frames <10% error': r'frames that has <10% error:\s*([0-9.]+)',
        'Frames <20% error': r'frames that has <20% error:\s*([0-9.]+)'
    }
    for line in lines:
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                metrics[key].append(float(match.group(1)))
    # Only include non-empty arrays
    filtered_metrics = {k: v for k, v in metrics.items() if v}
    return pd.DataFrame(filtered_metrics)

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

    def plot_test_loss_avg_frame(self, title=None, figsize=(8, 5), xlabel='Epoch',
                                 ylabel='Test loss avg over frame', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Test loss avg over frame'], label=name)
            plt.xticks(range(0, len(df['Test loss avg over frame']), xtick_step), fontsize=fontsize)
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

    def plot_frames_lt_5_error(self, title=None, figsize=(8, 5), xlabel='Epoch',
                               ylabel='Frames <5% error', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Frames <5% error'], label=name)
            plt.xticks(range(0, len(df['Frames <5% error']), xtick_step), fontsize=fontsize)
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

    def plot_frames_lt_10_error(self, title=None, figsize=(8, 5), xlabel='Epoch',
                                ylabel='Frames <10% error', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Frames <10% error'], label=name)
            plt.xticks(range(0, len(df['Frames <10% error']), xtick_step), fontsize=fontsize)
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

    def plot_frames_lt_20_error(self, title=None, figsize=(8, 5), xlabel='Epoch',
                                ylabel='Frames <20% error', save_path=None, xtick_step=5, fontsize=16):
        plt.figure(figsize=figsize)
        for name, df in self.log_dataframes.items():
            plt.plot(df['Frames <20% error'], label=name)
            plt.xticks(range(0, len(df['Frames <20% error']), xtick_step), fontsize=fontsize)
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

    # ET_32m_df = extract_dataframe_from_log(ET_dir / 'ET_hard_train_log.txt')
    # ET_32m_more_data_df = extract_dataframe_from_log(ET_dir / 'ET_32m_more_data_train_log.txt')
    # ET_plotter = LogMetricsPlotter_labels([('ET', ET_32m_df), ('ET+', ET_32m_more_data_df)])
    # ET_plotter.plot_loss(save_path=current_dir / 'ET hard' / 'test loss.pdf')
    # ET_plotter.plot_accuracy_per_frame(save_path=current_dir / 'ET hard' / 'accuracy per frame.pdf')

    AT_fc_embed_df = extract_dataframe_from_log(AT_dir / 'model2_fc_embed_20epoch_train_log.txt')
    AT_lin_embed_df = extract_dataframe_from_log(AT_dir / 'model2_lin_embed_train_log.txt')
    AT_plotter = LogMetricsPlotter_labels([('FC', AT_fc_embed_df),('Lin', AT_lin_embed_df)])
    AT_plotter.plot_test_loss_avg_frame(save_path= current_dir / 'AT training' / 'test loss avg over frame.pdf')
    AT_plotter.plot_frames_lt_20_error(save_path= current_dir / 'AT training' / 'Proportion of frames that has lt20% error.pdf')
    AT_plotter.plot_frames_lt_10_error(save_path= current_dir / 'AT training' / 'Proportion of frames that has lt10% error.pdf')
    AT_plotter.plot_frames_lt_5_error(save_path= current_dir / 'AT training' / 'Proportion of frames that has lt5% error.pdf')







