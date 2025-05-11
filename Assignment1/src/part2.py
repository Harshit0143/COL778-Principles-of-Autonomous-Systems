from utils import run_kalman_filter, Scene3D, get_xyz, get_Xt, plot_uncertainty_ellipse, gaussian_prob, Graph2D, generate_latex_table
import numpy as np
np.random.seed(0)

class FootballKinematics2D:
    def get_t_steps(self):
        return self.t_steps
    
    def get_BS_obs(self, Xt):
        Zt_ideal = np.linalg.norm(Xt[: 2].T[0] - self.points, axis = 1)[:, np.newaxis]
        noise = np.random.multivariate_normal([0] * self.points.shape[0], self.Qt_BS)[:, np.newaxis]
        return Zt_ideal + noise

    def get_Ct_BS(self, mu_bar_t):
        vector = mu_bar_t[: 2].T[0] - self.points
        dist = np.linalg.norm(vector, axis = 1)[:, np.newaxis]
        return np.hstack((vector / dist, np.zeros((self.points.shape[0], 2)))), dist



    def __init__(self, t_start, delta_t, num_steps, R, sigma_BS, X_0, Sigma_0, points):

        self.points = points
        self.Rt = R
        self.X_0, self.Sigma_0 = X_0, Sigma_0
        I2, O2 = np.identity(2), np.zeros((2, 2))
        self.At = np.block([
                [I2, delta_t * I2],
                [O2, I2]
            ])
        self.Bt = np.zeros((4, 1))
        self.Qt_BS = (sigma_BS ** 2) * np.identity(self.points.shape[0])
        self.t_steps = np.linspace(t_start, t_start + (num_steps - 1) * delta_t, num_steps)
        self.ut = [0.0] * len(self.t_steps)

        self.Xt = get_Xt(X_0 = self.X_0, 
                        t_steps = self.t_steps, 
                        At = self.At,
                        Bt = self.Bt,
                        acc_fn = lambda t: 0.0,
                        Rt = self.Rt) 
        self.zt = [self.get_BS_obs(Xt) for Xt in self.Xt]


    def get_trajectory_gt(self):
        return {
            'x': [Xt[0, 0].item() for Xt in self.Xt],
            'y': [Xt[1, 0].item() for Xt in self.Xt]
        }


    def get_trajectory(self):
      
        mu_t, Sigma_t, _ = run_kalman_filter(X0 = self.X_0,
                            Sigma0 = self.Sigma_0,
                            At = self.At,
                            Bt = self.Bt,
                            Rt = self.Rt,
                            Ct_fn = self.get_Ct_BS,
                            Qt = self.Qt_BS,
                            ut = self.ut,
                            zt = self.zt)
        return {
            'x': [mu[0, 0].item() for mu in mu_t],
            'y': [mu[1, 0].item() for mu in mu_t],
            'cov_xy': [Sigma[:2, :2] for Sigma in Sigma_t]
        }
    
    def uncertainty_ellipse(self, save_idx):
        bs_name = ''
        for point in self.points:
            bs_name += f'({int(point[0])}, {int(point[1])})'
        size = (4.2, 6)
        trajectory = self.get_trajectory_gt()
        x_gt, y_gt = trajectory['x'], trajectory['y']
        cov_gt = [np.zeros((2, 2))] * len(x_gt)
        plot_uncertainty_ellipse(x_gt, y_gt, cov_gt, label = f'Ground Truth {bs_name}', save_name = f'plots/part2-f-{save_idx}-GT.png', mark_points = True, size=size)

        trajectory = self.get_trajectory()
        x_fl, y_fl, cov_fl =  trajectory['x'], trajectory['y'], trajectory['cov_xy']
        return plot_uncertainty_ellipse(x_fl, y_fl, cov_fl, label = f'Filter BS: {bs_name}' ,save_name = f'plots/part2-f-{save_idx}-filter.png', mark_points = True, size=size)
        




"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
class FootballKinematics3D:
    def get_IMU_obs(self, Xt):
        Zt_ideal = Xt
        noise = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], self.Qt['IMU'])[:, np.newaxis]
        return Zt_ideal + noise
    
    def get_GPS_obs(self, Xt):
        Zt_ideal = Xt[: 3]
        noise = np.random.multivariate_normal([0, 0, 0], self.Qt['GPS'])[:, np.newaxis]
        return Zt_ideal + noise
    
    def get_BS_obs(self, Xt):
        Zt_ideal = np.linalg.norm(Xt[: 3].T[0] - self.points, axis = 1)[:, np.newaxis]
        noise = np.random.multivariate_normal([0] * self.points.shape[0], self.Qt['BS'])[:, np.newaxis]
        return Zt_ideal + noise

    def get_Ct_BS(self, mu_bar_t):
        vector = mu_bar_t[: 3].T[0] - self.points
        dist = np.linalg.norm(vector, axis = 1)[:, np.newaxis]
        return np.hstack((vector / dist, np.zeros((self.points.shape[0], 3)))), dist

    def get_Ct_IMU(self, mu_bar_t):
        Ct_IMU = np.identity(6)
        return Ct_IMU, Ct_IMU @ mu_bar_t
    
    def get_Ct_GPS(self, mu_bar_t):
        Ct_GPS = np.hstack((np.identity(3), np.zeros((3, 3)))) 
        return Ct_GPS, Ct_GPS @ mu_bar_t
        


    def __init__(self, t_start, delta_t, num_steps, R, sigma_GPS, sigma_BS, sigma_IMU, X_0, Sigma_0):

        self.points = np.array([
                            [32.0, -50.0, 10.0],
                            [-32.0, -50.0, 10.0],
                            [-32.0, 50.0, 10.0],
                            [32.0, 50.0, 10.0]
                        ])
        self.Rt = R
        self.g = -10.0
        # [xt yt zt x.t y.t z.t
        self.X_0, self.Sigma_0 = X_0, Sigma_0
        I3, O3 = np.identity(3), np.zeros((3, 3))
        self.At = np.block([
                [I3, delta_t * I3],
                [O3, I3]
            ])
        self.Bt = np.array([
                    [0.0],
                    [0.0],
                    [0.5 * (delta_t ** 2)],
                    [0.0],
                    [0.0],
                    [delta_t]
                ])
        # GPS: x y z, BS d1 d2 d3 d4, IMU x y z x. v. z.

        self.Qt = {
            'GPS' : (sigma_GPS ** 2) * I3,
            'IMU' : (sigma_IMU ** 2) * np.identity(6),
            'BS'  : (sigma_BS ** 2)  * np.identity(self.points.shape[0])
        }


             
            
    
        t_steps = np.linspace(t_start, t_start + (num_steps - 1) * delta_t, num_steps)
        self.ut = [self.g] * len(t_steps)
        self.Xt = get_Xt(X_0 = self.X_0, 
                        t_steps = t_steps, 
                        At = self.At,
                        Bt = self.Bt,
                        acc_fn = lambda t: self.g,
                        Rt = self.Rt) 

        self.Ct = {
            'BS': self.get_Ct_BS,
            'IMU': self.get_Ct_IMU,
            'GPS': self.get_Ct_GPS
        }

        self.zt = {
            'BS' : [self.get_BS_obs(Xt) for Xt in self.Xt],
            'IMU': [self.get_IMU_obs(Xt) for Xt in self.Xt],
            'GPS': [self.get_GPS_obs(Xt) for Xt in self.Xt]
        }


    def get_trajectory_gt(self):
        return {
            'x': [Xt[0, 0].item() for Xt in self.Xt],
            'y': [Xt[1, 0].item() for Xt in self.Xt],
            'z': [Xt[2, 0].item() for Xt in self.Xt]
        }


    def get_trajectory(self, sensor):

        mu_t, Sigma_t, _ = run_kalman_filter(X0 = self.X_0,
                            Sigma0 = self.Sigma_0,
                            At = self.At,
                            Bt = self.Bt,
                            Rt = self.Rt,
                            Ct_fn = self.Ct[sensor],
                            Qt = self.Qt[sensor],
                            ut = self.ut,
                            zt = self.zt[sensor])
        return {
            'x': [mu[0, 0].item() for mu in mu_t],
            'y': [mu[1, 0].item() for mu in mu_t],
            'z': [mu[2, 0].item() for mu in mu_t],
            'cov_xyz': [Sigma[:3, :3] for Sigma in Sigma_t]
        }
        
    
    def is_goal(self, trajectory, covar = None):
        for i in range(len(trajectory) - 1):
            x1, y1, z1 = trajectory[i]
            x2, y2, z2 = trajectory[i + 1]
            if y1 <= 50 <= y2 or y2 <= 50 <= y1:
                t = (50 - y1) / (y2 - y1)
                x_50 = (1 - t) * x1 + t * x2 
                z_50 = (1 - t) * z1 + t * z2
                if covar is not None:
                    covar_50 = (1 - t) * covar[i] + t * covar[i + 1]
                    sigma_x, sigma_z = np.sqrt(covar_50[0, 0].item()), np.sqrt(covar_50[2, 2].item())
                    px = gaussian_prob(mean = x_50, std_dev = sigma_x, x_start = -4, x_end = 4)
                    pz = gaussian_prob(mean = z_50, std_dev = sigma_z, x_start = 0, x_end = 3)
                    return px * pz > 0.8
                
                return -4 < x_50 < 4 and 0 < z_50 < 3
        return False

    def hit_goal_simulation(self):
        xyz_gt =  [get_xyz(Xt) for Xt in self.Xt]
        xyz_IMU = [get_xyz(obs_IMU) for obs_IMU in self.zt['IMU']]
        xyz_GPS = [get_xyz(obs_GPS) for obs_GPS in self.zt['GPS']]

        belief_GPS = self.get_trajectory('GPS')
        xyz_filter_GPS = [(belief_GPS['x'][i], belief_GPS['y'][i], belief_GPS['z'][i]) for i in range(len(belief_GPS['x']))]

        belief_IMU = self.get_trajectory('IMU')
        xyz_filter_IMU = [(belief_IMU['x'][i], belief_IMU['y'][i], belief_IMU['z'][i]) for i in range(len(belief_IMU['x']))]
        return {
            'Ground Truth': self.is_goal(xyz_gt),
            'IMU': self.is_goal(xyz_IMU),
            'GPS': self.is_goal(xyz_GPS),
            'Filter_{IMU}': self.is_goal(xyz_filter_IMU, covar = belief_IMU['cov_xyz']),
            'Filter_{GPS}': self.is_goal(xyz_filter_GPS, covar = belief_GPS['cov_xyz'])
        }
    
    def uncertainty_ellipse_GPS(self):
        size = (3.5, 6.5)
        trajectory = self.get_trajectory('GPS')
        x_fl, y_fl, cov_fl =  trajectory['x'], trajectory['y'], trajectory['cov_xyz']
        cov_fl = [Sigma[: 2, : 2] for Sigma in cov_fl]
        plot_uncertainty_ellipse(x_fl, y_fl, cov_fl, label = 'filter' ,save_name = 'plots/part2-e-filter.png', size = size)
        
        
        trajectory = [get_xyz(obs_GPS) for obs_GPS in self.zt['GPS']]
        x_GPS, y_GPS = zip(*[(p[0], p[1]) for p in trajectory])
        cov_GPS = [self.Qt['GPS'][: 2, : 2]] * len(x_GPS)
        plot_uncertainty_ellipse(x_GPS, y_GPS, cov_GPS, label = 'GPS', save_name = 'plots/part2-e-GPS.png', size = size)
        
        trajectory = self.get_trajectory_gt()
        x_gt, y_gt = trajectory['x'], trajectory['y']
        cov_gt = [np.zeros((2, 2))] * len(x_gt)
        plot_uncertainty_ellipse(x_gt, y_gt, cov_gt, label = 'Ground Truth', save_name = 'plots/part2-e-GT.png', size = size)




def run_simulations(params_dict, num_simulations):
    goals_cnt = {
        'Ground Truth': 0,
        'IMU': 0,
        'GPS': 0,
        'Filter_{IMU}': 0,
        'Filter_{GPS}': 0
    }
    for _ in range(num_simulations):
        ball = FootballKinematics3D(**params_dict)
        is_goal = ball.hit_goal_simulation()
        for key in is_goal:
            goals_cnt[key] += is_goal[key]

    return goals_cnt

def run_sims_params(params, var_params):
    print(var_params)
    Rt = np.diag([var_params['sigma_x'] ** 2] * 3 + [var_params['sigma_v'] ** 2] * 3)
    params['R'] = Rt
    params['sigma_GPS'] = var_params['sigma_s']
    return run_simulations(params_dict=params, num_simulations=1000)


            

if __name__ == '__main__':
    default_params = params = {
                        't_start': 0.0,
                        'delta_t': 0.01,
                        'num_steps': 130,
                        'R': np.diag([0.01 ** 2, 0.01 ** 2, 0.01 ** 2,
                                    0.1 ** 2, 0.1 ** 2, 0.1 ** 2]),
                        'sigma_GPS': 0.1,
                        'sigma_BS': 0.1,
                        'sigma_IMU': 0.1,
                        'X_0': np.array([[24.0], [4.0], [0.0], [-16.04], [36.8], [8.61]]),
                        'Sigma_0': 1e-4 * np.identity(6)
                }
    ball = FootballKinematics3D(**default_params)
    ##### part b
    for sensor in ['IMU', 'BS', 'GPS']:
        scene = Scene3D()
        trajectory = ball.get_trajectory_gt()
        scene.add((trajectory['x'], trajectory['y'], trajectory['z']), label = 'Ground Truth', color='blue')
        trajectory = ball.get_trajectory(sensor)
        scene.add((trajectory['x'], trajectory['y'], trajectory['z']), label=sensor, color='red')
        scene.save(f'plots/part2-b-{sensor}.html')



    ##### part c
    params = default_params.copy()
    params['num_steps'] = 150
    # run_simulations(params_dict=params, num_simulations=1000)
    ##### part d
    params = default_params.copy()

    sigma_x_vals = []
    sigma_v_vals = []
    sigma_s_vals = []
    # Uncomment below to get results. Takes time.
    # sigma_x_vals = [0.001, 0.01, 0.1]
    # sigma_v_vals = [0.01, 0.1, 1.0]
    # sigma_s_vals = [0.01, 0.1, 1.0]
    results_list = []
    for sigma_x in sigma_x_vals:
        for sigma_v in sigma_v_vals:
            for sigma_s in sigma_s_vals:
                var_params = {'sigma_x': sigma_x, 'sigma_v': sigma_v, 'sigma_s': sigma_s, 'num_steps': 150}
                goals = run_sims_params(params, var_params)
                results_list.append((var_params, goals))

    print(generate_latex_table(results_list))


    ##### part e
    ball.uncertainty_ellipse_GPS()

    ##### part f
    BS_set = [
        np.array([[-32.0, 50.0]]),
        np.array([[-32.0, -50.0]]),
        np.array([[-32.0, 0.0]]),
        np.array([
                [-32.0, 0.0],
                [32.0, 0.0]
            ])
        ]
    default_params = {
        't_start': 0.0,
        'delta_t': 0.01,
        'num_steps': 250,
        'R': np.diag([0.01 ** 2, 0.01 ** 2, 0.1 ** 2, 0.1 ** 2]),
        'sigma_BS': 0.1,
        'X_0': np.array([[0.0], [-50.0], [0.0], [40.0]]),
        'Sigma_0': np.identity(4),
        'points': None
    }
    ### part f.1
    params = default_params.copy()
    
    for idx in range(len(BS_set)):
        params['points'] = BS_set[idx]
        ball = FootballKinematics2D(**params)
        minor, major = ball.uncertainty_ellipse(save_idx = idx)
        scene = Graph2D(x_label = '$t$', y_label = r'$length$', mark_lines=False)
        scene.add((ball.get_t_steps(), minor), color = 'blue', label = r'$minor$ $axis$', linewidth = 1.5)
        scene.add((ball.get_t_steps(), major), color = 'red', label = r'$major$ $axis$', linewidth = 1.5)
        scene.save(f'plots/part2-f-{idx}-axes.png')

    






    