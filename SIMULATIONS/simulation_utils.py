import numpy as np
from scipy.stats import norm, gamma


def make_simulated_data(simulation_type, n_samples, signal_strength, seed):
    np.random.seed(seed+1)


    # simulation 1
    if simulation_type == 'linear':
        # confound noise term
        C_0 = np.random.normal(loc=0, scale=1, size=n_samples)
        # confound weight
        w_cy = 2.5


        # function with which y enters c
        def f(x):
            return x

        # feature noise term
        X_0 = np.random.normal(loc=0, scale=1, size=n_samples)
        w_xy = 0
        w_xc = 1

        # function with which y enters X
        def g(x):
            return x


        # function with which c enters X
        def h(x):
            return x

    # simulation 2
    elif simulation_type == 'nonlinear_3':
        C_0 = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        #C_0 = np.random.normal(loc=0, scale=0.6, size=n_samples)

        X_0 = np.random.normal(loc=0, scale=0.4, size=n_samples)

        w_cy = 1
        w_xy = 0
        w_xc = 1

        def f(x):
            out = []
            for val in x:
                if val == 0:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.1, 0.8, 0.1])[0]
                elif val == 1:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.4, 0.2, 0.4])[0]
                out.append(c)
            return out

        def g(x):
            return x

        def h(x):
            out = []
            for val in x:
                if val < 0.5 or val > 1.5:
                    out.append(0)
                else:
                    out.append(1)
            return np.array(out)

    # simulation 3
    elif simulation_type == 'nonlinear_4_noise':
        C_0 = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        X_0 = 0 #np.random.normal(loc=0, scale=0.4, size=n_samples)

        w_cy = 1
        w_xy = 0
        w_xc = 1

        def f(x):
            out = []
            for val in x:
                if val == 0:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.1, 0.8, 0.1])[0]
                elif val == 1:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.4, 0.2, 0.4])[0]
                out.append(c)
            return out

        def g(x):
            return x

        def h(x):
            out = []
            for val in x:
                if val < 0.5 or val > 1.5:
                    out.append(0 + np.random.normal(loc=0, scale=0.1))
                else:
                    out.append(1 + np.random.normal(loc=0, scale=0.6))
            return np.array(out)

    # simulation 4
    elif simulation_type == 'joint_XOR':
        C_0 = np.random.normal(loc=0, scale=1, size=n_samples)
        X_0 = 0

        w_cy = 2
        w_xy = 0  # all information in goes via the confound
        w_xc = 1


        def f(x):
            # TODO could be a degraded version of Y
            return x


        def g(x):
            return np.hstack([x.reshape(-1, 1), x.reshape(-1, 1)])


        def h(x):
            samples = []
            for val in x:
                if val >= 1:
                    cov = [[1, 0.9], [0.9, 1]]

                elif val < 1:
                    cov = [[1, -0.9], [-0.9, 1]]

                sample = np.random.multivariate_normal([0, 0], cov, 1)
                samples.append(sample)
            return np.vstack(samples)

    # simulation 5
    elif simulation_type == 'joint_AND':
        C_0 = np.random.normal(loc=0, scale=0.5, size=n_samples)
        X_0 = 0
        w_cy = 2
        w_xy = 0
        w_xc = 1

        def f(x):
            return x

        def g(x):
            return np.hstack([x.reshape(-1, 1), x.reshape(-1, 1)])


        def h(x):
            samples = []
            for val in x:
                cov = [[0.5, 0], [0, 0.5]]
                if val >= 1:
                    mu = [5, 5]
                    sample = np.random.multivariate_normal(mu, cov, 1)

                elif val < 1:
                    mu = np.random.choice([[0, 5], [5, 0]], p=[0.5, 0.5])
                    sample = np.random.multivariate_normal(mu, cov, 1)

                samples.append(sample)
            return np.vstack(samples)


        def h(x):
            samples = []
            for val in x:
                cov = [[0.5, 0], [0, 0.5]]
                if val >= 1:
                    mu = [5, 5]
                elif val < 1:
                    mu = np.array([[0, 5], [5, 0]])[np.random.randint(2), :]
                sample = np.random.multivariate_normal(mu, cov, 1)
                samples.append(sample)
            return np.vstack(samples)

    # simulation 6
    elif simulation_type == 'linear_true_2':
        C_0 = np.random.normal(loc=0, scale=0.6, size=n_samples)
        X_0 = np.random.normal(loc=0, scale=0.6, size=n_samples)

        w_cy = 0
        w_xy = signal_strength
        w_xc = 1


        def f(x):
            return x


        def g(x):
            return x


        def h(x):
            return x

    # sinmulation 7
    elif simulation_type == 'nonlinear_5_noise':
        C_0 = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        X_0 = np.random.normal(loc=0, scale=0.4, size=n_samples)

        w_cy = 1
        w_xy = signal_strength
        w_xc = 1

        def f(x):
            out = []
            for val in x:
                if val == 0:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.15, 0.7, 0.15])[0]
                elif val == 1:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.35, 0.3, 0.35])[0]
                out.append(c)
            return out

        def g(x):
            return 1-x

        def h(x):
            out = []
            for val in x:
                if val < 0.5 or val > 1.5:
                    out.append(0)
                else:
                    out.append(1)
            return np.array(out)

    # simulation 8
    elif simulation_type == 'joint_XOR_true_2':
        C_0 = np.random.normal(loc=0, scale=1, size=n_samples)
        X_0 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_samples)

        w_cy = 0
        w_xy = signal_strength
        w_xc = 1

        def f(x):
            return x

        def g(x):
            samples = []
            for val in x:
                if val == 1:
                    cov = [[1, 0.9], [0.9, 1]]

                elif val == 0:
                    cov = [[1, -0.9], [-0.9, 1]]
                sample = np.random.multivariate_normal([0, 0], cov, 1)
                samples.append(sample)
            return np.vstack(samples)


        def h(x):
            return np.hstack([x.reshape(-1, 1), x.reshape(-1, 1)])

    else:
        raise ValueError

    # generate simulation
    y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    C = C_0 + w_cy * f(y)
    X = X_0 + w_xy * g(y) + w_xc * h(C)

    try:
        X.shape[1]
    except IndexError:
        X = X.reshape(-1, 1)

    try:
        C.shape[1]
    except IndexError:
        C = C.reshape(-1, 1)

    return X, y, C


def make_simulated_data_highdim(simulation_type, n_samples, seed, n_feature,
                                noise_features=1, noise_confound=1):

    np.random.seed(seed+n_feature) # this ensures that C is different even if y is the same
                                   # in the nonlinear simulations

    # simulation 9
    if simulation_type == 'linear':
        # confound noise term
        C_0 = np.random.normal(loc=0, scale=noise_confound, size=n_samples)
        # confound weight
        w_cy = 2.5

        # function with which y enters c
        def f(x):
            return x

        # feature noise term
        X_0 = np.random.normal(loc=0, scale=noise_features, size=n_samples)
        w_xy = 0
        w_xc = 1

        # function with which y enters X
        def g(x):
            return x


        # function with which c enters X
        def h(x):
            return x

    # simulation 10
    elif simulation_type == 'nonlinear':
        C_0 = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        X_0 = np.random.normal(loc=0, scale=noise_features, size=n_samples)

        w_cy = 1
        w_xy = 0
        w_xc = 1

        def f(x):
            out = []
            for val in x:
                if val == 0:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.1, 0.8, 0.1])[0]
                elif val == 1:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.4, 0.2, 0.4])[0]
                out.append(c)
            return out

        def g(x):
            return x

        def h(x):
            out = []
            for val in x:
                if val < 0.5 or val > 1.5:
                    out.append(0)
                else:
                    out.append(1)
            return np.array(out)

    # simulation 11
    elif simulation_type == 'nonlinear_noise':
        C_0 = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        X_0 = 0

        w_cy = 1
        w_xy = 0
        w_xc = 1

        def f(x):
            out = []
            for val in x:
                if val == 0:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.1, 0.8, 0.1])[0]
                elif val == 1:
                    c = np.random.choice(a=[0, 1, 2], size=1, p=[0.4, 0.2, 0.4])[0]
                out.append(c)
            return out

        def g(x):
            return x

        def h(x):
            out = []
            for val in x:
                if val < 0.5 or val > 1.5:
                    out.append(0 + np.random.normal(loc=0, scale=0.1))
                else:
                    out.append(1 + np.random.normal(loc=0, scale=noise_features))
            return np.array(out)

    else:
        raise ValueError

    # generate simulation
    y = np.random.default_rng(seed=seed).choice([0, 1], size=n_samples, p=[0.5, 0.5])
    C = C_0 + w_cy * f(y)
    X = X_0 + w_xy * g(y) + w_xc * h(C)

    try:
        X.shape[1]
    except IndexError:
        X = X.reshape(-1, 1)

    try:
        C.shape[1]
    except IndexError:
        C = C.reshape(-1, 1)

    return X, y, C


