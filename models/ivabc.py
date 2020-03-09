import numpy as np

from utils import select_prototypes


class IVABC:
    def __init__(self, n_particles=25, n_prototypes=None, k=1, alpha=0.5, max_iter=200):
        self.n_particles_ = n_particles
        self.n_prototypes_ = n_prototypes
        self.k_ = k
        self.alpha_ = alpha
        self.max_iter_ = max_iter

    def fit(self, X, y):
        self.mins_ = X[:,::2].min(axis=0)
        self.maxs_ = X[:,1::2].max(axis=0)

        self.initialize_(X, y)

        t = 0  
        rep = 0
        f = self.pbest_fitness[self.arg_gbest]
        while t < self.max_iter_ and rep < 50:
            t += 1

            self.send_workers(X, y)
            self.send_onlookers(X, y, t)
            self.send_scouts(X, y)
            
            f_new = self.pbest_fitness[self.arg_gbest]
            if f_new == f:
                rep += 1
            else:
                rep = 0      
            f = f_new
        return self 

    def initialize_(self, X, y):
        self.limits_ = np.zeros(self.n_particles_)
        self.dropped_prots_ = np.zeros(
            (self.n_particles_, np.sum(self.n_prototypes_))
        )
        self.particles_ = []
        self.particle_y_ = []
        self.velocities_ = []
        self.particle_weights_ = []
        self.dropped_prots_ = []
        self.fitness_ = np.zeros(self.n_particles_)

        for particle in range(self.n_particles_):
            solution = generate_particle_(
                X, y, self.mins_, self.maxs_,
                self.n_prototypes_, self.k_
            )
            self.particles_.append(solution[0])
            self.particle_y_.append(solution[1])
            self.velocities_.append(
                np.random.uniform(
                    -1, 1, 
                    (np.sum(self.n_prototypes_) + 1, X.shape[1])
                )
            )
            self.particle_weights_.append(solution[2])
            self.dropped_prots_.append(solution[3])
            self.fitness_[particle] = calculate_fitness_(
                X, y, solution[0], solution[1],
                solution[2], solution[3], self.alpha_,
                self.maxs_ - self.mins_, self.k_
            )

        self.pbest, self.pbest_y = self.particles_[:], self.particle_y_[:]
        self.pbest_weights = self.particle_weights_[:]
        self.pbest_dropped = self.dropped_prots_[:]
        self.pbest_fitness = np.copy(self.fitness_)
        self.arg_gbest = self.pbest_fitness.argmin()
        
    def send_workers(self, X, y):
        p = int(X.shape[1] / 2)
        n_prots = np.sum(self.n_prototypes_)

        for particle in range(self.n_particles_):
            dropped = self.dropped_prots_[particle]
            j = np.random.choice(p)
            prot = np.random.choice(
                np.append(
                    np.delete(
                        np.arange(n_prots),
                        np.where(dropped == True)[0]
                    ) + 1, 0
                )
            )
            particle_k = np.random.choice(
                np.delete(
                    np.arange(self.n_particles_), particle
                )
            )
            
            phi = 2 * np.random.rand() - 1

            if prot != 0:

                previous_min = self.particles_[particle][prot, 2 * j]
                previous_max = self.particles_[particle][prot, 2 * j + 1]

                self.particles_[particle][prot, 2 * j] += (
                    phi * (
                        self.particles_[particle][prot, 2 * j
                        ] - self.particles_[particle_k][prot, 2 * j]
                    )
                )
                self.particles_[particle][prot, 2 * j + 1] += (
                    phi * (
                        self.particles_[particle][prot, 2 * j + 1
                        ] - self.particles_[particle_k][prot, 2 * j + 1]
                    )
                )

            else:
                previous_min = self.particles_[particle][prot, j]
                self.particles_[particle][prot, j] += (
                    phi * (
                        self.particles_[particle][prot, j
                        ] - self.particles_[particle_k][prot, j]
                    )
                )

            self.particles_[particle] = fix_intervals_(
                fix_kept_features_(
                    self.particles_[particle]
                ),
                self.mins_,
                self.maxs_
            ) 

            w = self.particle_weights_[particle]
            if prot == 0:
                w = None

            w, part, d_p = update_feature_weights_(
                X, y, self.particles_[particle], 
                self.particle_y_[particle], self.mins_, 
                self.maxs_, self.k_, 
                dropped_prots=self.dropped_prots_[particle], 
                feature_weights=w
            )

            f = calculate_fitness_(
                X, y, part, self.particle_y_[particle], w,
                d_p, self.alpha_, self.maxs_ - self.mins_,
                self.k_,
            )

            if f < self.fitness_[particle]:
                self.fitness_[particle] = f
                self.particles_[particle] = part
                self.particle_weights_[particle] = w
                self.dropped_prots_[particle] = d_p
                self.limits_[particle] = 0
            else:
                if prot != 0:
                    self.particles_[particle][prot, 2 * j] = previous_min 
                    self.particles_[particle][prot, 2 * j + 1] = previous_max 
                else:
                    self.particles_[particle][prot, j] = previous_min 
                self.limits_[particle] += 1

            if f < self.pbest_fitness[particle]:
                self.pbest_fitness[particle] = f
                self.pbest[particle] = np.copy(self.particles_[particle])
                self.pbest_weights[particle] = np.copy(self.particle_weights_[particle])  
                self.pbest_dropped[particle] = np.copy(self.dropped_prots_[particle])      
            
        self.arg_gbest = self.pbest_fitness.argmin()
    
    def send_onlookers(self, X, y, t):
        
        p = int(X.shape[1] / 2)
        
        inverse = 1.0 / (self.fitness_ + 1.0)
        probs = inverse / np.sum(inverse)
        
        w_min = 0.4
        w_max = 0.9
        
        inertia = w_min + (w_max - w_min) * (self.max_iter_ - t) / self.max_iter_
        c1 = 0.5 + np.random.rand() / 2.0
        c2 = 0.5 + np.random.rand() / 2.0

        for index in range(self.n_particles_):    
            particle = np.random.choice(int(self.n_particles_), p=probs)
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            self.velocities_[particle] = np.clip(
                inertia * self.velocities_[particle] + c1 * r1 * (
                    self.pbest[particle] - self.particles_[particle]
                ) + c2 * r2 * (
                    self.pbest[self.arg_gbest] - self.particles_[particle]
                ),
                -1, 1
            )

            kept_features = np.copy(self.particles_[particle][0,:p])  
            part = fix_intervals_(
                fix_kept_features_(
                    self.particles_[particle] + self.velocities_[particle]
                ),
                self.mins_,
                self.maxs_
            )   
            
            w = self.particle_weights_[particle]
            if np.any(np.around(kept_features) != np.around(part[0,:p])):
                w = None

            w, part, d_p = update_feature_weights_(
                X, y, part, 
                self.particle_y_[particle], self.mins_, 
                self.maxs_, self.k_, 
                dropped_prots=self.dropped_prots_[particle], 
                feature_weights=w
            )

            f = calculate_fitness_(
                X, y, part, self.particle_y_[particle], w,
                d_p, self.alpha_, self.maxs_ - self.mins_,
                self.k_
            )

            if f < self.fitness_[particle]:
                self.fitness_[particle] = f
                self.particles_[particle] = part
                self.particle_weights_[particle] = w
                self.dropped_prots_[particle] = d_p
                self.limits_[particle] = 0
            else:
                self.limits_[particle] += 1

            if f < self.pbest_fitness[particle]:
                self.pbest_fitness[particle] = f
                self.pbest[particle] = np.copy(self.particles_[particle])
                self.pbest_weights[particle] = np.copy(self.particle_weights_[particle])  
                self.pbest_dropped[particle] = np.copy(self.dropped_prots_[particle])      
            
            if f < self.pbest_fitness[self.arg_gbest]:
                self.arg_gbest = particle

    def send_scouts(self, X, y):
        
        for particle in np.arange(self.n_particles_)[self.limits_ >= 10]:
            solution = generate_particle_(
                X, y, self.mins_, self.maxs_,
                self.n_prototypes_, self.k_
            )
            self.particles_[particle] = solution[0]
            self.particle_y_[particle] = solution[1]
            self.velocities_[particle] = np.random.uniform(
                -1, 1, 
                (np.sum(self.n_prototypes_) + 1, X.shape[1])
            )
            self.particle_weights_[particle] = solution[2]
            self.dropped_prots_[particle] = solution[3]
            self.fitness_[particle] = calculate_fitness_(
                X, y, solution[0], solution[1],
                solution[2], solution[3], self.alpha_,
                self.maxs_ - self.mins_, self.k_
            )
                
            if self.fitness_[particle] < self.pbest_fitness[particle]:
                self.pbest_fitness[particle] = self.fitness_[particle]
                self.pbest[particle] = np.copy(self.particles_[particle])
                self.pbest_weights[particle] = np.copy(self.particle_weights_[particle])  
                self.pbest_dropped[particle] = np.copy(self.dropped_prots_[particle])       
            self.limits_[particle] = 0          
                
        self.arg_gbest = self.pbest_fitness.argmin()

    def predict(self, X):

        gbest = self.pbest[self.arg_gbest]
        prototypes_y = self.pbest_y[self.arg_gbest]
        feature_weights = self.pbest_weights[self.arg_gbest]
        amplitudes = self.maxs_ - self.mins_
        dropped_prots = self.pbest_dropped[self.arg_gbest]

        _, _, predictions = evaluate_(
            X, np.ones(len(X)), gbest, prototypes_y,
            feature_weights, dropped_prots,
            amplitudes, self.k_
        )

        return predictions

    def accuracy(self, X, y):
        predictions = self.predict(X)

        return np.mean(predictions == y)
        

def calculate_distances_(
    X, prototypes_X, weights, kept_features, amplitudes
):
    kf_bool = kept_features.astype(bool)

    w = weights[:, kf_bool]

    ampl = amplitudes[kf_bool]

    mins_prots, mins_X = (
        prototypes_X[:,::2][:, kf_bool], X[:,::2][:, kf_bool]
    )
    maxs_prots, maxs_X = (
        prototypes_X[:,1::2][:, kf_bool], X[:,1::2][:, kf_bool]
    )

    d_mins = ((
        ((mins_prots - mins_X[:, np.newaxis]) / ampl) ** 2
    ) * w).sum(axis=2)
    d_maxs = ((
        ((maxs_prots - maxs_X[:, np.newaxis]) / ampl) ** 2
    ) * w).sum(axis=2)
    
    return (d_mins + d_maxs) / (2 * kf_bool.sum())


def fix_kept_features_(particle):
    p = int(particle.shape[1] / 2)
    particle[0,:p] = np.clip(particle[0,:p], 0, 1)
    while np.all(np.around(particle[0,:p]) == 0):
        particle[0,:p] = np.random.rand(p)
    return particle


def fix_intervals_(particle, mins, maxs):
    ind = particle[1:,::2] > particle[1:,1::2]
    if np.any(ind):
        temp = particle[1:,::2][ind]
        particle[1:,::2][ind] = particle[1:,1::2][ind]
        particle[1:,1::2][ind] = temp
    particle[1:,::2] = np.clip(particle[1:,::2], mins, maxs)
    particle[1:,1::2] = np.clip(particle[1:,1::2], mins, maxs)
    return particle


def update_feature_weights_(
    X, y, particle, prototypes_y, mins, maxs, 
    k, dropped_prots=None, feature_weights=None
):
    amplitudes = maxs - mins
    n = len(X)
    n_labels = len(np.unique(prototypes_y))
    p = int(particle.shape[1] / 2) 
    n_prots = len(prototypes_y)

    kept_features = np.around(particle[0,:p])
    kept_p = kept_features.sum()
    
    if dropped_prots is None:
        dropped_prots = np.zeros(n_prots, dtype=bool)
    if feature_weights is None:
        feature_weights = np.ones((n_prots, p))

    prototypes_X = particle[1:]

    distances = calculate_distances_(
        X, prototypes_X, feature_weights, kept_features, amplitudes
    )
    
    distances[:,dropped_prots] = np.inf
    k_neighbors = np.argsort(distances, axis=1)[
        :,:min(k, n_prots-np.sum(dropped_prots))
    ]

    k_dists = distances[np.arange(n)[:,None], k_neighbors]
    correct = (prototypes_y[k_neighbors] == y[:,None])
    k_dist_correct = correct * k_dists
    
    n_correct = np.sum(correct, axis=1)
    total_correct = np.sum(n_correct > 0)
    n_correct[n_correct == 0] = 1
    
    dists_prots = np.array(
        [np.sum(
            k_dist_correct[k_neighbors == prot]
        ) for prot in np.arange(n_prots)]
    )
    temp = ((2 * correct - 1) * (k_neighbors + 1)).ravel()
    occur = np.bincount(temp[temp >= 0], minlength=n_prots + 1)[1:]
    class_info = np.array(
        [
            [
                np.sum(dists_prots[prototypes_y == c]),
                np.sum((dists_prots > 0).__and__(prototypes_y == c)), 
                np.prod(
                    dists_prots[
                        (dists_prots > 0).__and__(prototypes_y == c)
                    ] / occur[
                        (dists_prots > 0).__and__(prototypes_y == c)
                    ]
                )
            ] for c in range(n_labels)
        ]
    )
    
    class_totals = class_info[:,0] / class_info[:,1]
    class_totals[np.isnan(class_totals)] = 0.0
    if np.any(class_totals == 0):
        class_weights = np.ones(n_labels)
        for c in range(n_labels):
            if class_totals[c] == 0.0:
                c_members = prototypes_y == c
                (
                    prototypes_X[c_members,:], 
                    _
                ) = select_prototypes(
                    X[y == c], y[y == c], [np.sum(c_members)] * n_labels
                )
                dropped_prots[c_members] = False
                feature_weights[c_members,:] = 1.0
    else:
        class_weights = (class_totals.prod() ** (1 / n_labels)) / class_totals        
    
    prot_weights = np.ones(n_prots)   
    ind_prots = np.arange(n_prots)[np.logical_not(dropped_prots)]
    prod_classes = class_info[:,2]
    for prot in ind_prots:
        y_prot = int(prototypes_y[prot])
        if class_totals[y_prot] > 0.0:
            if dists_prots[prot] == 0:
                    dropped_prots[prot] = True
            else:
                num = (
                    class_weights[y_prot] * prod_classes[y_prot]
                ) ** (1 / class_info[y_prot,1])
                prot_weights[prot] = num / (dists_prots[prot] / occur[prot])
                members = np.where((k_neighbors == prot).__and__(correct))[0]
                mi = ((
                    prototypes_X[prot,::2] - X[members,::2]
                ) / (amplitudes)) ** 2
                ma = ((
                    prototypes_X[prot,1::2] - X[members,1::2]
                ) / (amplitudes)) ** 2
                diff = (mi + ma) / (2 * kept_features.sum())

                delta = np.sum(
                    kept_features * (
                        diff / n_correct[members, None] / total_correct
                    ),
                    axis=0
                )
                flag = np.any((kept_features == 1.0).__and__(delta == 0.0))
                if flag == False:
                    feature_weights[prot,:] = np.array([
                        ((
                            prot_weights[prot] * np.prod(delta[delta > 0])
                        ) ** (1 / kept_p)) / v if v > 0 else 1 for v in delta
                    ])
                else:
                    feature_weights[prot,:] = np.array(
                        [
                            prot_weights[prot] ** (
                                1 / kept_p
                            ) if v > 0 else 1 for v in kept_features
                        ]
                    )
    particle[1:] = prototypes_X
    return feature_weights, particle, dropped_prots


def generate_particle_(X, y, mins, maxs, n_prots, k):
    kept_features = np.random.rand(len(maxs) * 2)    
    prototypes_X, prototypes_y = select_prototypes(X, y, n_prots)
    particle = fix_intervals_(
        fix_kept_features_(
            np.vstack([kept_features, prototypes_X])
        ),
        mins,
        maxs
    ) 
    feature_weights, particle, dropped_prots = update_feature_weights_(
        X, y, particle, prototypes_y, mins, maxs, k
    )
    return particle, prototypes_y, feature_weights, dropped_prots


def evaluate_(
    X, y, particle, prototypes_y, feature_weights, dropped_prots,
    amplitudes, k

):
    n = len(X)
    p = int(particle.shape[1] / 2) 
    kept_features = np.around(particle[0,:p])
    prototypes_X = particle[1:]
    n_labels = len(np.unique(prototypes_y))
    n_prots = len(prototypes_y)

    distances = calculate_distances_(
        X, prototypes_X, feature_weights, kept_features, amplitudes
    )
    
    distances[:,dropped_prots] = np.inf
    k_neighbors = np.argsort(distances, axis=1)[
        :,:min(k, n_prots-np.sum(dropped_prots))
    ]

    k_dists = distances[np.arange(n)[:,None], k_neighbors]

    inverse = 1.0 / (k_dists + 1e-100)
    members = np.array([prototypes_y[k_neighbors] == c for c in range(n_labels)])
    omegas = np.sum(members * inverse, axis=2).T

    winners = np.argmax(omegas, axis=1)
    ind_zero = np.where(k_dists == 0.0)[0]
    for ind in ind_zero:
        kd = k_dists[ind,:]
        if np.sum(kd == 0.0) == 1:
            winners[ind] = prototypes_y[k_neighbors[ind, 0]]
        else:
            winners[ind] = np.argmax(
                np.bincount(
                    prototypes_y[k_neighbors[ind, kd == 0]], minlength=n_labels
                )
            )
    
    correct = (prototypes_y[k_neighbors] == y[:,None])
    k_dist_correct = correct * k_dists

    n_correct = np.sum(correct, axis=1)
    total_correct = np.sum(n_correct > 0)
    n_correct[n_correct == 0] = 1

    criterion = np.sum(k_dist_correct / n_correct[:,None]) / total_correct
    
    return np.mean(winners != y), criterion, winners


def calculate_fitness_(
    X, y, particle, prototypes_y, feature_weights, dropped_prots, alpha,
    amplitudes, k
):
    error, criterion, _ = evaluate_(
        X, y, particle, prototypes_y, feature_weights, dropped_prots,
        amplitudes, k
    )

    return alpha * error + (1 - alpha) * criterion
 