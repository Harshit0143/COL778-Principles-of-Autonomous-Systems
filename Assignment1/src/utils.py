import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.patches import Ellipse
from math import erf, sqrt



def get_Xt(X_0, t_steps, At, Bt, acc_fn, Rt):
    Xt = [X_0]
    curr = X_0
    
    for t in t_steps[: -1]:
        curr = At @ curr + Bt * acc_fn(t)
        curr += np.random.multivariate_normal([0] * X_0.shape[0], Rt)[:, np.newaxis]
        Xt.append(curr)
    return Xt


class Scene3D:
    def __init__(self):
        self.fig = go.Figure()
        self._plot_field()

    def _plot_field(self):
        field_corners = [(32, -50, 0), (-32, -50, 0), (-32, 50, 0), (32, 50, 0), (32, -50, 0)]
        goal_corners = [(-4, 50, 0), (-4, 50, 3), (4, 50, 3), (4, 50, 0), (-4, 50, 0)]
        
        field_x, field_y, field_z = zip(*field_corners)
        goal_x, goal_y, goal_z = zip(*goal_corners)

        self.fig.add_trace(go.Scatter3d(x=field_x, y=field_y, z=field_z, mode='lines', name='Field', line=dict(color='green', width=3)))
        self.fig.add_trace(go.Scatter3d(x=goal_x, y=goal_y, z=goal_z, mode='lines', name='Goal Post', line=dict(color='red', width=5)))
        max_range = max(35, 55, 4)
        self.fig.update_layout(
            title='',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[-max_range, max_range]),  
                yaxis=dict(range=[-max_range, max_range]),
                zaxis=dict(range=[-max_range, max_range]),
                aspectmode='cube'
            ),
            showlegend=True
        )

    def add(self, trajectory, label, color):
        if not trajectory:
            return
        x, y, z = trajectory
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name= label, line=dict(color=color, width=3)))

    def save(self, save_path):
        pio.write_html(self.fig, save_path)





def kalman_update(At, Bt, Rt, Ct_fn, Qt,  mu, Sigma, ut, zt, obs_valid):
    '''
    Ct has shape [m, n]
    mu has shape [n, 1]
    zt has shape [m, 1]
    Sigma has shape [n, n]
    Kt has shape [n, m]
    '''
    mu_bar_t = At @ mu + Bt * ut # [n, 1]
    sigma_bar_t = At @ Sigma @ At.T + Rt
    if not obs_valid:
        return mu_bar_t, sigma_bar_t, np.zeros((mu.shape[0], zt.shape[0]))
        # This K will not be used later. Just here for consistency.
    Ct, h_mu_bar_t = Ct_fn(mu_bar_t)
    Kt = sigma_bar_t @ Ct.T @ np.linalg.inv(Ct @ sigma_bar_t @ Ct.T + Qt) # [n, m]
    mu_t = mu_bar_t + Kt @ (zt - h_mu_bar_t)
    sigma_t = (np.identity(mu_t.shape[0]) - Kt @ Ct) @ sigma_bar_t

    return mu_t, sigma_t, Kt

def run_kalman_filter(X0, Sigma0, At, Bt, Rt, Ct_fn, Qt, ut, zt, obs_valid = None):
    # print('Running Kalman Filer...')
    assert len(ut) == len(zt)
    if obs_valid is None:
        obs_valid = [True] * len(zt)
    mu, Sigma = X0, Sigma0
    mu_t, Sigma_t, Kt = [mu], [Sigma], []
    for idx in range(1, len(ut)):
        mu, Sigma, K = kalman_update(At = At,
                                Bt = Bt,
                                Rt = Rt,
                                Ct_fn = Ct_fn,
                                Qt = Qt,
                                mu = mu,
                                Sigma = Sigma,
                                ut = ut[idx - 1],
                                zt = zt[idx],
                                obs_valid = obs_valid[idx])
        mu_t.append(mu)
        Sigma_t.append(Sigma)
        Kt.append(K)
    return mu_t, Sigma_t, Kt

def get_xyz(state):
    return state[0, 0].item(), state[1, 0].item(), state[2, 0].item()


def gaussian_prob(mean, std_dev, x_start, x_end):
    def gaussian_cdf(x):
        return 0.5 * (1 + erf((x - mean) / (std_dev * sqrt(2))))
    return gaussian_cdf(x_end) - gaussian_cdf(x_start)


class Graph2D:
    def __init__(self, x_label, y_label, title='', mark_lines = True):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.trajectories = []
        self.vlines = []
        if mark_lines:
            self.add_vline(0.25, label = r'$t = 0.25$')
            self.add_vline(3.00, label = r'$t = 3.00$')


    def add(self, trajectory, color='b', label=None, linewidth=1.0):
        x_vector, y_vector = trajectory
        self.trajectories.append((x_vector, y_vector, label, color, linewidth))

    def plot_heatmap(self, trajectory, label=None, color='b', alpha=0.3, linewidth=1.0):
        x_vector, y_vector, sigma_y = trajectory
        self.trajectories.append((x_vector, y_vector, sigma_y, label, color, alpha, linewidth))

    def add_vline(self, x_value, color='r', linestyle='dotted', linewidth=1.5, label=None):
        '''Adds a vertical line at x_value.'''
        self.vlines.append((x_value, color, linestyle, linewidth, label))

    def save(self, save_path, dpi=1000):
        plt.figure(figsize=(8, 5))

        # Plot trajectories
        for item in self.trajectories:
            if len(item) == 5:  # Standard trajectory
                x_vector, y_vector, label, color, linewidth = item
                plt.plot(x_vector, y_vector, color=color, label=label, linewidth=linewidth)
            elif len(item) == 7:  # Heatmap with uncertainty
                x_vector, y_vector, sigma_y, label, color, alpha, linewidth = item
                plt.plot(x_vector, y_vector, color=color, label=label, linewidth=linewidth)
                plt.fill_between(x_vector, y_vector - sigma_y, y_vector + sigma_y, color=color, alpha=alpha, edgecolor='none')

        # Plot vertical lines
        for x_value, color, linestyle, linewidth, label in self.vlines:
            plt.axvline(x=x_value, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

        plt.xlabel(self.x_label, fontsize=16)
        plt.ylabel(self.y_label, fontsize=16)
        plt.title(self.title)

        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend()

        plt.grid()
        plt.savefig(save_path, dpi=1000)
        plt.close()


def plot_uncertainty_ellipse(x_list, y_list, cov_matrices, label, save_name, mark_points=False, size = (8, 8)):
    fig, ax = plt.subplots(figsize=size)
    ax.plot(x_list, y_list, color="green", label=label, linewidth=0.5, alpha=1.0, zorder=1)
    minor_axes, major_axes = [], []
    for i in range(len(x_list)):  
        mean = (x_list[i], y_list[i])
        cov = cov_matrices[i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        width, height = 2 * np.sqrt(eigvals)
        major_axes.append(max(width, height))
        minor_axes.append(min(width, height))
        ellipse = Ellipse(mean, width, height, angle=np.degrees(angle),
                          facecolor='purple', edgecolor='none', alpha=1.0, zorder=2)  
        ax.add_patch(ellipse)
    
    if mark_points:
        points = {
            'B1': [-32, 50],
            'B2': [-32, -50],
            'B3': [-32, 0],
            'B4': [32, 0]
        }
        x_coords = [point[0] for point in points.values()]
        y_coords = [point[1] for point in points.values()]
        ax.scatter(x_coords, y_coords, color='red', zorder=3, s = 5)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('$y_t$')
    ax.legend(fontsize=10, loc='upper right')
    
    plt.savefig(save_name, dpi=1000)
    plt.close()
    return minor_axes, major_axes


def generate_latex_table(results_list):
    latex_str = '''\\begin{table}[H]
    \\centering
    \\begin{tabular}{|c|c|c|c|c|c|c|c|}
        \\hline
        $\\sigma_x, \\sigma_y, \\sigma_z$ & $\\sigma_{\\dot{x}}, \\sigma_{\\dot{y}}, \\sigma_{\\dot{z}}$ & $\\sigma_{BS}$ & Ground Truth & IMU & Filter$_{IMU}$ & GPS & Filter$_{GPS}$ \\\\
        \\hline
    '''
    for params, goals_cnt in results_list:
        latex_str += f"        ${params['sigma_x']:.4f}$ & ${params['sigma_v']:.4f}$ & ${params['sigma_s']:.4f}$ & "
        latex_str += f"${goals_cnt['Ground Truth']}$ & ${goals_cnt['IMU']}$ & ${goals_cnt['Filter_{IMU}']}$ & ${goals_cnt['GPS']}$ & ${goals_cnt['Filter_{GPS}']}$ \\\\\n"

    latex_str += '''        \\hline
    \\end{tabular}
    \\caption{Goal detection results for different parameter values.}
    \\label{tab:goal_results}
    \\end{table}
    '''
    return latex_str