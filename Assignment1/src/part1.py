from utils import run_kalman_filter, Graph2D, get_Xt
import numpy as np

np.random.seed(42)

class TrainKinematics:
    def get_t_steps(self):
        return self.t_steps
    def get_acc(self, t):
        if t < 0.25:
            return 400.00
        elif 3 <= t and t < 3.25:
            return -400.00
        return 0.00
        

    
    def get_obs(self, Xt):
        Zt_ideal = (2 * Xt[:1, :] / self.v_sound)
        noise = np.random.multivariate_normal([0], self.Qt)[:, np.newaxis]
        return Zt_ideal + noise
    
    def get_Ct(self, mu_bar_t):
        return self.Ct, self.Ct @ mu_bar_t

    def __init__(self, v_sound, t_start,  delta_t, num_steps, R, sigma_s, X_0, Sigma_0):
        self.Rt = R
        self.Qt = np.array([[sigma_s ** 2]])
        self.v_sound = v_sound
        self.X_0, self.Sigma_0 = X_0, Sigma_0
        self.At = np.array([
                    [1.0, delta_t], 
                    [0.0, 1.0]
                ])
        self.Bt = np.array([
                    [0.5 * (delta_t ** 2)],
                    [delta_t]
                ])
        self.Ct = np.array([[2 / v_sound, 0]])
        self.t_steps = np.linspace(t_start, t_start + (num_steps - 1) * delta_t, num_steps)
        self.ut = [self.get_acc(t) for t in self.t_steps]
        self.Xt = get_Xt(X_0 = self.X_0, 
                        t_steps = self.t_steps, 
                        At = self.At,
                        Bt = self.Bt,
                        acc_fn = self.get_acc,
                        Rt = self.Rt) 
        self.zt = [self.get_obs(Xt) for Xt in self.Xt]

       

    def get_states(self):
        pos, vel= [Xt[0, 0].item() for Xt in self.Xt], [Xt[1, 0].item() for Xt in self.Xt]
        return pos, vel
        
    def get_belief(self, t_start = np.inf, t_end = -np.inf):
        obs_valid = [not (t_start <= t <= t_end) for t in self.t_steps]
        mu_t, var_t, Kt = run_kalman_filter(X0 = self.X_0,
                                    Sigma0 = self.Sigma_0,
                                    At = self.At,
                                    Bt = self.Bt,
                                    Rt = self.Rt,
                                    Ct_fn = self.get_Ct,
                                    Qt = self.Qt,
                                    ut = self.ut,
                                    zt = self.zt,
                                    obs_valid = obs_valid)

        xe, ve = [mu[0, 0].item() for mu in mu_t], [mu[1, 0].item() for mu in mu_t]
        xe_stdev, ve_stdev = [np.sqrt(var_t[0, 0]).item() for var_t in var_t], [np.sqrt(var_t[1, 1]).item() for var_t in var_t]
        kx, kv = [K[0, 0].item() for K in Kt], [K[1, 0].item() for K in Kt]

        return {
            'xe': np.array(xe),
            've': np.array(ve),
            'xe_stdev': np.array(xe_stdev),
            've_stdev': np.array(ve_stdev),
            'kalman_gain_x': np.array(kx),
            'kalman_gain_v': np.array(kv)
        }
    
    

  

def plot_part_d(train1, train2, train3, idx):
    scene = Graph2D(x_label = '$t$', y_label =  r'$x^e_t$', mark_lines = False)
    t_steps = train1[0].get_t_steps()
    belief1 = train1[0].get_belief()
    belief2 = train2[0].get_belief()
    belief3 = train3[0].get_belief()

    scene.plot_heatmap((t_steps, belief1['xe'], belief1['xe_stdev']), color = 'red', label = train1[1], linewidth = 0.5)
    scene.plot_heatmap((t_steps, belief2['xe'], belief2['xe_stdev']), color = 'blue', label = train2[1], linewidth = 0.5)
    scene.plot_heatmap((t_steps, belief3['xe'], belief3['xe_stdev']), color = 'green', label = train3[1], linewidth = 0.5)
    scene.save(f'plots/part1-d.{idx}-xe.png')

    linewidth = 1.0
    scene = Graph2D(x_label = '$t$', y_label = r'$x_t$', mark_lines = False)
    xt1, _ = train1[0].get_states()
    xt2, _ = train2[0].get_states()
    xt3, _ = train3[0].get_states()
    scene.add((t_steps, xt1), color = 'red', label = train1[1], linewidth = linewidth)
    scene.add((t_steps, xt2), color = 'blue', label = train2[1], linewidth = linewidth)
    scene.add((t_steps, xt3), color = 'green', label = train3[1], linewidth = linewidth)
    scene.save(f'plots/part1-d.{idx}-xt.png')

    scene = Graph2D(x_label = '$t$', y_label = r'$K^x_t$', mark_lines = False)
    scene.add((t_steps[1: ], belief1['kalman_gain_x']), color = 'red', label = train1[1], linewidth = linewidth)
    scene.add((t_steps[1: ], belief2['kalman_gain_x']), color = 'blue', label = train2[1], linewidth = linewidth)
    scene.add((t_steps[1: ], belief3['kalman_gain_x']), color = 'green', label = train3[1], linewidth = linewidth)
    scene.save(f'plots/part1-e.{idx}-kx.png')

    scene = Graph2D(x_label = '$t$', y_label = r'$K^{\dot{x}}_t$', mark_lines = False)
    scene.add((t_steps[1: ], belief1['kalman_gain_v']), color = 'red', label = train1[1], linewidth = linewidth)
    scene.add((t_steps[1: ], belief2['kalman_gain_v']), color = 'blue', label = train2[1], linewidth = linewidth)
    scene.add((t_steps[1: ], belief3['kalman_gain_v']), color = 'green', label = train3[1], linewidth = linewidth)
    scene.save(f'plots/part1-e.{idx}-kv.png')



if __name__ == '__main__':
    default_params = {
        'v_sound': 3000,
        't_start': 0.0,
        'delta_t': 0.01,
        'num_steps': 326,
        'R': np.diag([0.1 ** 2, 0.5 ** 2]),
        'sigma_s': 0.01,
        'X_0': np.array([[0.0],
                         [0.0]]),
        'Sigma_0': 1e-4 * np.identity(2)
    }

    train = TrainKinematics(**default_params)
    ###### part a
    t_steps = train.get_t_steps()
    xt, vt = train.get_states()
    scene = Graph2D(x_label = '$t$', y_label = '$x_t$', mark_lines = False)
    scene.add((t_steps, xt), color = 'blue')
    scene.save('plots/part1-a.1.png')

    scene = Graph2D(x_label = '$t$', y_label = r'$\dot{x}_t$', mark_lines = False)
    scene.add((t_steps, vt), color = 'blue')
    scene.save('plots/part1-a.2.png')

    ###### part b
    belief = train.get_belief()

    scene = Graph2D(x_label = '$t$', y_label = r'$x^e_t$', mark_lines = False)
    scene.add((t_steps, belief['xe']), color = 'blue')
    scene.save('plots/part1-b.1.png')

    scene = Graph2D(x_label = '$t$', y_label = r'$\dot{x}^e_t$', mark_lines = False)
    scene.add((t_steps, belief['ve']), color = 'blue')
    scene.save('plots/part1-b.2.png')

    ##### part c
    scene = Graph2D(x_label = '$t$', y_label = r'$x$', mark_lines = False)
    scene.add((t_steps, xt), color = 'blue', label = '$x_t$', linewidth = 0.5)
    scene.plot_heatmap((t_steps, belief['xe'], belief['xe_stdev']), color = 'red', label = '$x^e_t$', linewidth = 0.5)
    scene.save('plots/part1-c.png')

    #### part d and e
    ## varying sigma_x
    params = default_params.copy()
    sigma_1, sigma_2, sigma_3 = 0.01, 0.1, 1.0
    params['R'] = np.diag([sigma_1 ** 2, 0.5 ** 2])
    train_1 = TrainKinematics(**params)
    params['R'] = np.diag([sigma_2 ** 2, 0.5 ** 2])
    train_2 = TrainKinematics(**params)
    params['R'] = np.diag([sigma_3 ** 2, 0.5 ** 2])
    train_3 = TrainKinematics(**params)
    plot_part_d((train_1, r'$\sigma_x = $' + f'{sigma_1:0.2f}'),
                (train_2, r'$\sigma_x = $' + f'{sigma_2:0.2f}'),
                (train_3, r'$\sigma_x = $' + f'{sigma_3:0.2f}'), 0)
    
    ## varying sigma_x_dot
    sigma_1, sigma_2, sigma_3 = 0.05, 0.5, 5.0
    params['R'] = np.diag([0.1 ** 2, sigma_1 ** 2])
    train_1 = TrainKinematics(**params)
    params['R'] = np.diag([0.1 ** 2, sigma_2 ** 2])
    train_2 = TrainKinematics(**params)
    params['R'] = np.diag([0.1 ** 2, sigma_3 ** 2])
    train_3 = TrainKinematics(**params)
    plot_part_d((train_1, r'$\sigma_{\dot{x}} = $' + f'{sigma_1:0.2f}'),
                (train_2, r'$\sigma_{\dot{x}} = $' + f'{sigma_2:0.2f}'),
                (train_3, r'$\sigma_{\dot{x}} = $' + f'{sigma_3:0.2f}'), 1)

    # varying sigma_s
    params = default_params.copy()
    sigma_1, sigma_2, sigma_3 = 0.001, 0.01, 0.1
    params['sigma_s'] = sigma_1
    train_1 = TrainKinematics(**params)
    params['sigma_s'] = sigma_2
    train_2 = TrainKinematics(**params)
    params['sigma_s'] = sigma_3
    train_3 = TrainKinematics(**params)
    plot_part_d((train_1, r'$\sigma_s = $' + f'{sigma_1:0.2f}'),
                (train_2, r'$\sigma_s = $' + f'{sigma_2:0.2f}'),
                (train_3, r'$\sigma_s = $' + f'{sigma_3:0.2f}'), 2)
    

    ### part f

    belief = train.get_belief(t_start=1.5, t_end=2.5)
    scene = Graph2D(x_label = '$t$', y_label = r'$x$', mark_lines = False)
    scene.add_vline(1.5, label = r'$t = 1.5$')
    scene.add_vline(2.5, label = r'$t = 2.5$')
    scene.plot_heatmap((t_steps, belief['xe'], belief['xe_stdev']), color = 'red', label = '$x^e_t$', linewidth = 0.5)
    scene.add((t_steps, xt), color = 'blue', label = '$x_t$', linewidth = 0.5)
    scene.save('plots/part1-f.png')

